# SPDX-License-Identifier: MIT
"""Weight-only 8-bit quantization (w8a16) for PersonaPlex/Moshi.

Conservative alternative to fp8_quantize: weights are *stored* as
float8_e4m3fn with a per-output-channel scale, dequantized in-kernel, and all
compute (accumulation and activations) stays bf16/fp32 — activation precision
is untouched. The matmul is a hand-written Triton GEMV that streams the fp8
weight matrix once and dequantizes in registers, so the memory-bandwidth win
of 8-bit weights is kept without an fp8 matmul or activation scaling.

Same integration contract as fp8_quantize:

    lm = loaders.get_moshi_lm(...)
    quantize_model_w8a16(lm)   # replace weights + patch forwards
    # ... warmup (CUDA graphs capture the Triton kernel) ...

Layer coverage mirrors fp8_quantize exactly (same min_features gate, same
depformer-self_attn skip, same bare in_proj handling) so the two schemes are
directly comparable in latency and divergence measurements.

The class-level forward patches dispatch on per-instance flags and also
honor fp8_quantize's `_is_fp8` markers, so both schemes can coexist in one
process (used by bench/divergence.py's three-way soak).
"""

import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# ============================================================================
# Triton GEMV: y[o] = (sum_i x[i] * W8[o,i]) * scale[o]   (bf16 x, fp8 W)
#
# Triton is imported lazily inside the builder so that `import moshi` and
# this module work on CPU-only / Triton-less installs; the kernel is built
# once and cached on first use.
# ============================================================================

_GEMV_KERNEL = None


def _build_gemv_kernel():
    global _GEMV_KERNEL
    if _GEMV_KERNEL is not None:
        return _GEMV_KERNEL
    import importlib
    g = globals()
    g["triton"] = importlib.import_module("triton")
    g["tl"] = importlib.import_module("triton.language")

    @triton.jit
    def _w8a16_gemv_kernel(
        x_ptr, w_ptr, scale_ptr, bias_ptr, y_ptr,
        IN: tl.constexpr, OUT,
        HAS_BIAS: tl.constexpr,
        BLOCK_OUT: tl.constexpr, BLOCK_IN: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_o = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        mask_o = offs_o < OUT
        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        for i in range(0, IN, BLOCK_IN):
            offs_i = i + tl.arange(0, BLOCK_IN)
            mask_i = offs_i < IN
            x = tl.load(x_ptr + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs_o[:, None] * IN + offs_i[None, :],
                        mask=mask_o[:, None] & mask_i[None, :],
                        other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        scale = tl.load(scale_ptr + offs_o, mask=mask_o, other=0.0)
        y = acc * scale
        if HAS_BIAS:
            y += tl.load(bias_ptr + offs_o, mask=mask_o,
                         other=0.0).to(tl.float32)
        tl.store(y_ptr + offs_o, y.to(y_ptr.dtype.element_ty), mask=mask_o)

    _GEMV_KERNEL = _w8a16_gemv_kernel
    return _GEMV_KERNEL


def w8a16_linear(x, w_fp8, scale, bias=None):
    """Weight-only-8bit linear. x: [..., IN] bf16; w_fp8: [OUT, IN] fp8e4m3;
    scale: [OUT] fp32. Compute fp32-accumulate, output bf16."""
    orig_shape = x.shape
    in_features = w_fp8.shape[1]
    out_features = w_fp8.shape[0]
    x2 = x.reshape(-1, in_features)
    if x2.shape[0] != 1:
        # Rare non-decode path (prompt batches etc.): correctness fallback,
        # full dequant + cuBLAS.
        w = (w_fp8.to(torch.float32) * scale[:, None]).to(x.dtype)
        return F.linear(x, w, bias).reshape(*orig_shape[:-1], out_features)
    y = torch.empty(1, out_features, device=x.device, dtype=x.dtype)
    # Tuned on GB10 (see M2 microbench): big shapes are bandwidth-bound and
    # favor long inner blocks; small depformer shapes favor more warps.
    if in_features >= 4096:
        BLOCK_OUT, BLOCK_IN, num_warps = 16, 512, 2
    else:
        BLOCK_OUT, BLOCK_IN, num_warps = 16, 256, 4
    kernel = _build_gemv_kernel()
    grid = ((out_features + BLOCK_OUT - 1) // BLOCK_OUT,)
    kernel[grid](
        x2, w_fp8, scale,
        bias if bias is not None else scale,  # dummy ptr when no bias
        y,
        IN=in_features, OUT=out_features,
        HAS_BIAS=bias is not None,
        BLOCK_OUT=BLOCK_OUT, BLOCK_IN=BLOCK_IN,
        num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], out_features)


# ============================================================================
# Forward patches (dispatch on instance flags; fp8-compatible, see module doc)
# ============================================================================

def _w8a16_forward(self, x):
    return w8a16_linear(x, self.weight, self.w8a16_scale, self.bias)


def _make_gating_forward():
    from .fp8_quantize import fp8_linear

    def _apply(lin, x):
        # Per-instance dispatch: this class-level patch may be installed
        # after fp8_quantize's, so it must recognize both schemes' markers.
        if getattr(lin, "_is_w8a16", False):
            return w8a16_linear(x, lin.weight, lin.w8a16_scale)
        if getattr(lin, "_is_fp8", False):
            return fp8_linear(x, lin.weight, lin.weight_scale)
        return F.linear(x, lin.weight)

    def gating_forward(self, x):
        x = _apply(self.linear_in, x)
        B, T, _ = x.shape
        x = x.view(B, T, 2, -1)
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        return _apply(self.linear_out, x)

    return gating_forward


def _make_attn_forward():
    from einops import rearrange as _rearrange
    from .fp8_quantize import fp8_linear

    def attn_forward(self, query, key, value):
        import moshi.modules.transformer as tf_mod

        state = self._streaming_state
        T = query.shape[1]
        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)
            offset_cpu = 0
        else:
            offset = state.offset
            offset_cpu = state.offset_cpu

        if self.weights_per_step:
            projected = tf_mod.multi_linear(
                self.weights_per_step, self.in_proj_weight, query, offset_cpu
            )
        elif getattr(self, "_in_proj_w8a16", False):
            projected = w8a16_linear(query, self._in_proj_q_weight,
                                     self._in_proj_q_scale)
        elif getattr(self, "_in_proj_fp8", False):
            projected = fp8_linear(query, self._in_proj_fp8_weight,
                                   self._in_proj_scale)
        else:
            projected = F.linear(query, self.in_proj_weight)

        q, k, v = _rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )
        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        k, v, pos_k = self._complete_kv(k, v)
        if self.causal:
            pos_k = pos_k.view(1, -1)
            pos_q = offset + torch.arange(
                T, device=query.device, dtype=torch.long
            ).view(-1, 1)
            delta = pos_q - pos_k
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = _rearrange(x, "b h t d -> b t (h d)")
        if self.weights_per_step:
            x = tf_mod.multi_linear(
                self.weights_per_step, self.out_proj.weight, x, offset_cpu
            )
        else:
            out_proj = self.out_proj
            if getattr(out_proj, "_is_w8a16", False):
                x = w8a16_linear(x, out_proj.weight, out_proj.w8a16_scale)
            elif getattr(out_proj, "_is_fp8", False):
                x = fp8_linear(x, out_proj.weight, out_proj.weight_scale)
            else:
                x = out_proj(x)
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x

    return attn_forward


# ============================================================================
# Weight quantization
# ============================================================================

def _quantize_weight(w):
    """Per-output-channel fp8e4m3 storage. Returns (w_fp8, scale_fp32)."""
    amax = w.abs().amax(dim=1).float()
    scale = (amax / 448.0).clamp(min=1e-12)
    w_fp8 = (w.float() / scale[:, None]).to(torch.float8_e4m3fn)
    return w_fp8, scale


def quantize_linear_w8a16(module):
    w_fp8, scale = _quantize_weight(module.weight.data)
    module.weight = nn.Parameter(w_fp8, requires_grad=False)
    module.register_buffer("w8a16_scale", scale)
    module._is_w8a16 = True
    module.forward = types.MethodType(_w8a16_forward, module)


def quantize_model_w8a16(model, min_features=512):
    """Quantize all large Linear layers to weight-only fp8 (bf16 compute).

    Coverage mirrors fp8_quantize.quantize_model: skips small linears and
    depformer self_attn; also converts main-attention bare in_proj_weight.
    """
    import importlib.util
    if not torch.cuda.is_available():
        raise RuntimeError("--w8a16 requires a CUDA device.")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError(
            "--w8a16 requires Triton for its dequant GEMV kernel; install "
            "it with `pip install triton`. No silent fallback is provided: "
            "without the kernel the flag cannot meet its performance "
            "contract.")
    import moshi.modules.gating as gating_mod
    import moshi.modules.transformer as tf_mod

    gating_mod.ActivationGating.forward = _make_gating_forward()
    tf_mod.StreamingMultiheadAttention.forward = _make_attn_forward()

    n_lin = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if module.in_features < min_features and module.out_features < min_features:
                continue
            if module.weight.ndim > 2:
                continue
            if "depformer" in name and "self_attn" in name:
                continue
            quantize_linear_w8a16(module)
            n_lin += 1

    n_inproj = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, tf_mod.StreamingMultiheadAttention):
            if "depformer" in name or module.weights_per_step:
                continue
            w = module.in_proj_weight.data
            if w.ndim != 2:
                continue
            w_fp8, scale = _quantize_weight(w)
            module.register_buffer("_in_proj_q_weight", w_fp8)
            module.register_buffer("_in_proj_q_scale", scale)
            module._in_proj_w8a16 = True
            # free the bf16 copy immediately (kernel never reads it)
            module.in_proj_weight = nn.Parameter(
                torch.empty(0, dtype=w.dtype, device=w.device),
                requires_grad=False,
            )
            n_inproj += 1

    torch.cuda.empty_cache()
    logger.info(f"[w8a16] Quantized {n_lin} Linear + {n_inproj} in_proj_weight")
    logger.info(f"[w8a16] GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model
