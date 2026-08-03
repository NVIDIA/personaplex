"""FP8 dynamic quantization for PersonaPlex/Moshi.

Usage:
    from moshi.fp8_quantize import quantize_model, free_bf16_inproj

    lm = loaders.get_moshi_lm(...)
    quantize_model(lm)       # Quantize weights + patch forward paths
    # ... warmup ...
    free_bf16_inproj(lm)     # Free original bf16 in_proj copies

Replaces nn.Linear weights with FP8 (float8_e4m3fn) and patches the
gating/attention forward paths to use torch._scaled_mm for native FP8 GEMM.

Benchmarked results:
    - Jetson Thor:  114ms → 74ms lm_step (1.54x), 16.7 → 11.2 GB
    - DGX Spark:    95.7ms → 70ms lm_step (1.37x), 18.8 → 11.0 GB
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import types

logger = logging.getLogger(__name__)


def fp8_linear(x, weight_fp8, weight_scale, bias=None):
    """FP8 GEMM with dynamic per-tensor input scaling."""
    orig_shape = x.shape
    in_features = weight_fp8.shape[1]
    out_features = weight_fp8.shape[0]
    x_2d = x.reshape(-1, in_features)

    x_amax = x_2d.abs().amax()
    x_scale = (x_amax / 448.0).clamp(min=1e-12)
    x_fp8 = (x_2d / x_scale).to(torch.float8_e4m3fn)

    out = torch._scaled_mm(
        x_fp8, weight_fp8.t(),
        scale_a=x_scale.float().view(1),
        scale_b=weight_scale,
        out_dtype=torch.bfloat16
    )

    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], out_features)


# ============================================================================
# Module-level forward patch for nn.Linear
# ============================================================================

def _fp8_forward(self, x):
    """Patched nn.Linear.forward using dynamic FP8."""
    return fp8_linear(x, self.weight, self.weight_scale, self.bias)


# ============================================================================
# Gating forward patch (ActivationGating uses F.linear, not nn.Linear.forward)
# ============================================================================

def _make_gating_forward():
    def gating_forward_fp8(self, x):
        lin_in = self.linear_in
        lin_out = self.linear_out
        if getattr(lin_in, '_is_fp8', False):
            x = fp8_linear(x, lin_in.weight, lin_in.weight_scale)
        else:
            x = F.linear(x, lin_in.weight)
        B, T, _ = x.shape
        x = x.view(B, T, 2, -1)
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        if getattr(lin_out, '_is_fp8', False):
            x = fp8_linear(x, lin_out.weight, lin_out.weight_scale)
        else:
            x = F.linear(x, lin_out.weight)
        return x
    return gating_forward_fp8


# ============================================================================
# Attention forward patch (in_proj_weight is bare nn.Parameter, not nn.Linear)
# ============================================================================

def _make_attn_forward():
    from einops import rearrange as _rearrange

    def attn_forward_fp8(self, query, key, value):
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
        else:
            if getattr(self, '_in_proj_fp8', False):
                projected = fp8_linear(query, self._in_proj_fp8_weight, self._in_proj_scale)
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
            if getattr(out_proj, '_is_fp8', False):
                x = fp8_linear(x, out_proj.weight, out_proj.weight_scale)
            else:
                x = out_proj(x)
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x

    return attn_forward_fp8


# ============================================================================
# Weight quantization
# ============================================================================

def quantize_linear_fp8(module):
    """Quantize a single nn.Linear to FP8 in-place."""
    w = module.weight.data
    amax = w.abs().amax()
    scale = (amax / 448.0).clamp(min=1e-12)
    module.weight = nn.Parameter(
        (w / scale).to(torch.float8_e4m3fn), requires_grad=False
    )
    module.register_buffer('weight_scale', scale.float().view(1))
    module._is_fp8 = True
    module.forward = types.MethodType(_fp8_forward, module)


def quantize_model(model, min_features=512):
    """Quantize all large Linear layers in the model to FP8.

    Patches ActivationGating and StreamingMultiheadAttention forward methods
    to use FP8 GEMM via torch._scaled_mm.

    Skips depformer self_attn (matrices too small — overhead > savings).
    """
    import moshi.modules.gating as gating_mod
    import moshi.modules.transformer as tf_mod

    gating_mod.ActivationGating.forward = _make_gating_forward()
    tf_mod.StreamingMultiheadAttention.forward = _make_attn_forward()

    linear_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if module.in_features < min_features and module.out_features < min_features:
                continue
            if module.weight.ndim > 2:
                continue
            # Skip depformer self_attn — per-step slices too small for FP8 benefit
            if 'depformer' in name and 'self_attn' in name:
                continue
            quantize_linear_fp8(module)
            linear_count += 1

    # Quantize bare in_proj_weight parameters on main transformer attention
    inproj_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, tf_mod.StreamingMultiheadAttention):
            if 'depformer' in name:
                continue
            if module.weights_per_step:
                continue
            w = module.in_proj_weight.data
            if w.ndim != 2:
                continue
            amax = w.abs().amax()
            scale = (amax / 448.0).clamp(min=1e-12)
            module.register_buffer('_in_proj_fp8_weight',
                (w / scale).to(torch.float8_e4m3fn))
            module._in_proj_scale = scale.float().view(1)
            module._in_proj_fp8 = True
            inproj_count += 1

    logger.info(f"[FP8] Quantized {linear_count} Linear + {inproj_count} in_proj_weight")
    logger.info(f"[FP8] GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


# ============================================================================
# Cleanup — free original bf16 copies after CUDA graph warmup
# ============================================================================

def free_bf16_inproj(model):
    """Free bf16 in_proj_weight copies after warmup (KV cache already allocated).

    Call this after state.warmup() — the CUDA graph has captured the FP8 path,
    so the original bf16 weights are no longer needed.
    """
    import moshi.modules.transformer as tf_mod
    freed = 0
    for name, module in model.named_modules():
        if isinstance(module, tf_mod.StreamingMultiheadAttention):
            if getattr(module, '_in_proj_fp8', False):
                sz = module.in_proj_weight.numel() * module.in_proj_weight.element_size()
                module.in_proj_weight = nn.Parameter(
                    torch.empty(0, dtype=torch.bfloat16, device=module.in_proj_weight.device),
                    requires_grad=False
                )
                freed += sz
    if freed > 0:
        torch.cuda.empty_cache()
        logger.info(f"[FP8] Freed {freed/1e9:.2f} GB bf16 in_proj_weight copies")
        logger.info(f"[FP8] GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
