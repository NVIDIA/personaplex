# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer model, with streaming support, + CUDA Graphable.
Optimized for inference.

See `StreamingTransformer` for more information.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import typing as tp

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..utils.compile import no_compile, torch_compile_lazy
from .gating import make_gating
from .rope import RotaryEmbedding
from .streaming import StreamingModule, StreamingContainer


class LayerNormF32(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_f32 = input.float()
        out_f32 = super().forward(x_f32)
        return out_f32.to(input.dtype)


def _rms_norm(
    x: torch.Tensor,
    alpha: torch.Tensor,
    dtype: tp.Optional[torch.dtype],
    eps: float,
):
    assert x.dim() == 3, f"RMSNorm expects 3D inputs but got {x.shape}"
    x_dtype = x.dtype
    if dtype is not None:
        x = x.to(dtype)
    var = eps + torch.mean(x**2, dim=2, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: tp.Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor):
        return _rms_norm(x, self.alpha, self.dtype, self.eps)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.
    """

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full(
                (channels,), init, requires_grad=True, device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module for transformer encoder layer.

    Args:
        norm_type (str): Normalization method.
        dim (int): Dimension of the normalized layer.
        **kwargs (dict): Additional parameters for normalization layer.
    Returns:
        nn.Module: Normalization module.
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type in {"rms_norm"}:
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full(
        [], max_period, device=positions.device, dtype=dtype
    )  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def multi_linear(
    num_linear: int,
    weight: torch.Tensor,
    x: torch.Tensor,
    offset: int,
):
    """Utility to apply a multi linear layer to the given input. A multi linear layer
    applies a different set of weight for each time step.

    Args:
        num_linear (int): Number of possible time steps and so number of linears.
        weight (torch.Tensor): Weight tensor, with shape `[num_linear * chout, chin]`.
        x (torch.Tensor): Input tensor, with shape `[B, T, C]`.
        offset (int): offset for the current time step, in particular for decoding, with
            time steps provided one by one.
    """
    B, T, C = x.shape
    ys = []
    chout, chin = weight.shape
    weight = weight.view(num_linear, -1, chin)
    for t in range(T):
        y = F.linear(x[:, t], weight[t + offset])
        ys.append(y)
    out = torch.stack(ys, 1)
    return out


def set_attention_context(model: nn.Module, context: tp.Optional[int] = None) -> None:
    """Deactivates or changes the context span (in time steps) in a model.
    Args:
        model (nn.Module): model over which to look for attentions.
        context (int or None): new temporary context value.

    ..Note:: this is not a context manager but a plain function changing the context forever.
        Initially, it was a context manager, but that led to interesting bugs when using
        activation checkpointing, with the context being inconsistent between the forward
        and backward.
    """
    for module in model.modules():
        if isinstance(module, StreamingMultiheadAttention):
            module.context = context


def enable_kv_prefix_pinning(model: nn.Module, max_pinned: int) -> int:
    """Make the causal RoPE self-attention layers of `model` use a `PinnedRingKVCache`.

    Must be called before streaming starts (`streaming()` / `streaming_forever()`), because the
    cache type is chosen when the streaming state is created. Layers without RoPE, without a
    finite context, or with per-step weights (e.g. the depformer) are left untouched.

    Returns:
        The number of attention layers that will pin.
    """
    count = 0
    for module in model.modules():
        if (
            isinstance(module, StreamingMultiheadAttention)
            and module.rope is not None
            and module.causal
            and module.context is not None
            and not module.weights_per_step
        ):
            if module._streaming_state is not None:
                raise RuntimeError(
                    "enable_kv_prefix_pinning must be called before streaming starts."
                )
            module.kv_pin_max = max_pinned
            count += 1
    return count


def pin_kv_prefix(model: nn.Module, n: int) -> int:
    """Pin the first `n` cached frames in every `PinnedRingKVCache` of `model`.

    Call outside CUDA graph capture, right after the prefix has been written.

    Returns:
        The number of attention layers that were pinned.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, StreamingMultiheadAttention):
            state = module._streaming_state
            if state is not None and isinstance(state.kv_cache, PinnedRingKVCache):
                state.kv_cache.pin(n)
                count += 1
    return count


class KVCacheResult(tp.NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions)


def _streaming_device(module: nn.Module, device: torch.device) -> torch.device:
    """Device for streaming state of `module`, given the device of its weights.

    With accelerate CPU offload the weights of an offloaded module are `meta`
    placeholders and the real device is the execution device of its hook; state
    allocated on `meta` cannot be used in the forward pass.
    """
    if device.type != "meta":
        return device
    hooks = [getattr(module, "_hf_hook", None)]
    while hooks:
        hook = hooks.pop(0)
        execution_device = getattr(hook, "execution_device", None)
        if execution_device is not None and not isinstance(execution_device, dict):
            return torch.device(execution_device)
        hooks.extend(getattr(hook, "hooks", None) or [])  # accelerate SequentialHook
    return device


class RingKVCache:
    """Efficient streaming KVCache to be compatible with Cuda Graph.

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of heads in the attention.
        dim_per_head (int): Dimension per head.
        device (torch.device): Device on which to initialize the cache.
        dtype (torch.dtype): dtype to use for the cache.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)

    def reset(self):
        self.end_offset.zero_()

    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype) + self.end_offset
        indexes = indexes % self.capacity
        self.cache[0].index_copy_(2, indexes, k)
        self.cache[1].index_copy_(2, indexes, v)
        self.end_offset.add_(T)

        keys = self.cache[0]
        values = self.cache[1]

        indexes = torch.arange(
            self.capacity, device=self.end_offset.device, dtype=torch.long
        )
        invalid = indexes >= self.end_offset

        end_index = self.end_offset % self.capacity
        delta = indexes - end_index

        # If last key is for step S, and capacity is C, last key was written at index S % C.
        # then end_offset = S + 1, and end_index = (S + 1) % C.
        # Then for index = (S % C), delta = -1, and the next code gives us:
        # position(index) = (S + 1) - 1 = S, all good.
        # Now the time step at end_offset is actually the oldest in the KVCache, e.g., its
        # position should be (S - self.capacity + 1).
        # The following code gives us:
        # position(index + 1) = S + 1 + 0 - self.capacity.

        positions = torch.where(
            delta <= 0,
            self.end_offset + delta,
            self.end_offset + delta - self.capacity,
        )
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)

    def asdict(self):
        return {"cache": self.cache, "end_offset": self.end_offset}


def _rotate_rope_keys(x: torch.Tensor, shift: torch.Tensor, max_period: float) -> torch.Tensor:
    """Move RoPE-encoded vectors forward by `shift` positions.

    `x` must already carry RoPE exactly as written by `apply_rope` (interleaved (real, imag)
    pairs on the last dimension, layout [B, H, T, D]). RoPE is a per-pair rotation, so rotating
    a key that was encoded at position p by `shift` yields the key encoded at p + shift; no
    un-rotated copy is needed. The frequencies are computed exactly as in `apply_rope`, and the
    angle is reduced modulo 2*pi in float64 so that very large shifts keep full precision.

    Compiled lazily like `apply_rope`, so the elementwise ops are fused into a few kernels
    (set NO_TORCH_COMPILE=1 to run it eagerly).

    Args:
        x: tensor of shape [B, H, T, D].
        shift: long tensor with one element. It may live on the device, so this function can be
            captured in a CUDA graph with a shift that changes at every replay.
        max_period: the `max_period` of the RotaryEmbedding that encoded `x`.
    """
    B, H, T, D = x.shape
    assert D % 2 == 0
    ds = torch.arange(D // 2, device=x.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    angle = torch.remainder(freqs.double() * shift.double(), 2 * math.pi)
    rotr = torch.cos(angle).float()
    roti = torch.sin(angle).float()

    xp = x.view(B, H, T, D // 2, 2)
    xr = xp[..., 0].float()
    xi = xp[..., 1].float()
    outr = xr * rotr - xi * roti
    outi = xr * roti + xi * rotr
    return torch.stack([outr.to(x.dtype), outi.to(x.dtype)], dim=-1).view(B, H, T, D)


rotate_rope_keys = torch_compile_lazy(_rotate_rope_keys)


@torch_compile_lazy
def _reencode_pinned_prefix(
    pinned_keys: torch.Tensor,
    head: torch.Tensor,
    n: torch.Tensor,
    shift: torch.Tensor,
    pin_slots: torch.Tensor,
    max_period: float,
) -> torch.Tensor:
    """New content for the first `max_pinned` key slots: the pinned keys moved by `shift` in
    slots < n, and the current content everywhere else. Fused into one kernel when compiled."""
    rotated = _rotate_rope_keys(pinned_keys, shift, max_period)
    keep = (pin_slots < n).view(1, 1, -1, 1)
    return torch.where(keep, rotated, head)


class PinnedRingKVCache:
    """Streaming KV cache whose first `n` entries (e.g. a system prompt) are never evicted.

    The `capacity` slots are split in two once `pin(n)` has been called:

    * slots [0, n): the pinned prefix. At every step its keys are re-encoded from a saved copy so
      that the prefix always sits immediately before the oldest remaining ring entry, i.e. at
      positions [E - capacity, E - capacity + n) once the ring has wrapped, where E is the number
      of frames seen so far. Every pinned key therefore stays within `capacity - 1` of the newest
      query: inside the attention context and inside the distance range seen in training. Before
      the ring wraps the shift is 0, so nothing differs from an ordinary cache.
    * slots [n, capacity): a FIFO ring holding the most recent `capacity - n` entries at their
      true positions.

    Values carry no positional information and are never modified after being written.

    With n == 0 (after construction or `reset()`) the cache returns exactly what `RingKVCache`
    returns, positions included, so it can be swapped in without changing any result until `pin`
    is called.

    Everything that runs per step uses device tensors and static shapes, so `complete` can be
    captured in a CUDA graph; `n` lives on the device and may change between sessions without
    re-capturing. The per-step cost of re-encoding is proportional to `max_pinned` (not `n`),
    because shapes must stay static for the graph. Only `pin` synchronizes with the host, and it
    must be called outside graph capture.

    Args:
        batch_size, num_heads, dim_per_head, capacity, device, dtype: as for `RingKVCache`.
        max_pinned (int): largest prefix that can be pinned; must be smaller than `capacity`.
        max_period (float): `max_period` of the RotaryEmbedding used by the attention.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        max_pinned: int,
        max_period: float,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        if not 0 < max_pinned < capacity:
            raise ValueError(
                f"max_pinned must be in (0, capacity={capacity}), got {max_pinned}."
            )
        self.capacity = capacity
        self.max_pinned = max_pinned
        self.max_period = max_period
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)
        self.pinned_n = torch.zeros(1, device=device, dtype=torch.long)
        # Keys of the pinned prefix as encoded at their original positions 0..n-1.
        self.pinned_keys = torch.zeros(
            (batch_size, num_heads, max_pinned, dim_per_head),
            device=device,
            dtype=dtype,
        )
        self._slots = torch.arange(capacity, device=device, dtype=torch.long)
        self._pin_slots = torch.arange(max_pinned, device=device, dtype=torch.long)

    def reset(self):
        self.end_offset.zero_()
        self.pinned_n.zero_()

    @torch.no_grad()
    def pin(self, n: int) -> None:
        """Pin the first `n` entries. Call right after they were written, before the ring wraps.

        Synchronizes with the host, so it must not be called during CUDA graph capture.
        """
        n = int(n)
        end = int(self.end_offset.item())
        if n < 0 or n > self.max_pinned:
            raise ValueError(
                f"Cannot pin {n} frames: at most {self.max_pinned} can be pinned "
                "(raise max_pinned / --max-pinned-frames)."
            )
        if n > end:
            raise ValueError(f"Cannot pin {n} frames, only {end} have been written.")
        if end > self.capacity:
            raise ValueError(
                "pin() must be called before the cache wraps around, while the first "
                f"entries are still present (end_offset={end}, capacity={self.capacity})."
            )
        if n == 0:
            self.pinned_n.zero_()
            return
        self.pinned_keys[:, :, :n].copy_(self.cache[0, :, :, :n])
        self.pinned_n.fill_(n)

    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        n = self.pinned_n
        ring = self.capacity - n

        # Write the new entries. With n == 0 this is `abs_pos % capacity`, as in RingKVCache.
        abs_pos = torch.arange(T, device=self.end_offset.device, dtype=torch.long) + self.end_offset
        slots = torch.where(abs_pos < n, abs_pos, n + torch.remainder(abs_pos - n, ring))
        self.cache[0].index_copy_(2, slots, k)
        self.cache[1].index_copy_(2, slots, v)
        self.end_offset.add_(T)
        end = self.end_offset

        # Re-encode the pinned prefix right before the oldest ring entry (no-op when n == 0).
        shift = torch.clamp(end - self.capacity, min=0)
        head = self.cache[0, :, :, : self.max_pinned]
        head.copy_(_reencode_pinned_prefix(
            self.pinned_keys, head, n, shift, self._pin_slots, self.max_period))

        # Positions exactly as RingKVCache computes them (used while nothing is pinned).
        all_slots = self._slots
        end_index = end % self.capacity
        delta = all_slots - end_index
        legacy = torch.where(delta <= 0, end + delta, end + delta - self.capacity)
        legacy = torch.where(all_slots >= end, torch.full_like(legacy, -1), legacy)

        # Positions with a pinned prefix: the newest ring entry has ring-relative index
        # `last_rel`; slot s holds the largest ring-relative index <= last_rel congruent to s - n.
        last_rel = end - 1 - n
        rel = last_rel - torch.remainder(last_rel - (all_slots - n), ring)
        ring_pos = torch.where(
            (all_slots >= n) & (rel >= 0), n + rel, torch.full_like(rel, -1)
        )
        pinned_pos = torch.where(all_slots < n, all_slots + shift, ring_pos)

        positions = torch.where(n > 0, pinned_pos, legacy)
        return KVCacheResult(self.cache[0], self.cache[1], positions)

    def asdict(self):
        return {
            "cache": self.cache,
            "end_offset": self.end_offset,
            "pinned_n": self.pinned_n,
            "pinned_keys": self.pinned_keys,
        }


@dataclass
class _MHAState:
    kv_cache: tp.Union[RingKVCache, PinnedRingKVCache]
    offset: torch.Tensor
    offset_cpu: int

    def reset(self):
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0


class StreamingMultiheadAttention(StreamingModule[_MHAState]):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Number of time steps the attention can access to.
            When causal, can access `context` time steps into the past, and when non causal,
            can access `context // 2` steps in the past, and the same in the future.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    _fsdp_final = True
    # When > 0, streaming allocates a PinnedRingKVCache that can pin up to this many frames.
    # Set it with `enable_kv_prefix_pinning` before streaming starts.
    kv_pin_max: int = 0

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        weights_per_step: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        out_dim = 3 * embed_dim
        mult = 1
        self.weights_per_step = weights_per_step
        if weights_per_step:
            mult = weights_per_step
        in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False, **factory_kwargs)
        # We try to follow the default PyTorch MHA convention, to easily compare results.
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(
            embed_dim, mult * embed_dim, bias=False, **factory_kwargs
        )

    def _init_streaming_state(self, batch_size: int) -> _MHAState:
        if self.context is None:
            if self.weights_per_step:
                capacity = self.weights_per_step
            else:
                raise RuntimeError(
                    "Cannot create a streaming KVCache without a context to estimate capacity."
                )
        else:
            capacity = self.context
        device = _streaming_device(self, self.in_proj_weight.device)
        # TODO: the following estimation will not work great with FSDP.
        dtype = self.in_proj_weight.dtype
        dim_per_head = self.embed_dim // self.num_heads
        kv_cache: tp.Union[RingKVCache, PinnedRingKVCache]
        if self.kv_pin_max > 0:
            assert self.rope is not None, "KV prefix pinning requires RoPE."
            kv_cache = PinnedRingKVCache(
                batch_size, self.num_heads, dim_per_head, capacity,
                max_pinned=self.kv_pin_max, max_period=self.rope.max_period,
                device=device, dtype=dtype,
            )
        else:
            kv_cache = RingKVCache(
                batch_size, self.num_heads, dim_per_head, capacity, device, dtype
            )
        return _MHAState(
            kv_cache,
            offset=torch.zeros(1, device=device, dtype=torch.long),
            offset_cpu=0,
        )

    def _complete_kv(self, k, v) -> KVCacheResult:
        state = self._streaming_state
        if state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            return state.kv_cache.complete(k, v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        state = self._streaming_state
        T = query.shape[1]

        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)
            offset_cpu = 0
        else:
            assert self.causal, "Streaming only available for causal"
            offset = state.offset
            offset_cpu = state.offset_cpu

        if self.weights_per_step:
            projected = multi_linear(
                self.weights_per_step, self.in_proj_weight, query, offset_cpu
            )
        else:
            projected = nn.functional.linear(query, self.in_proj_weight)
        q, k, v = rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )

        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        k, v, pos_k = self._complete_kv(k, v)
        if self.causal:
            pos_k = pos_k.view(1, -1)
            pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(
                -1, 1
            )
            delta = pos_q - pos_k
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        if self.weights_per_step:
            x = multi_linear(self.weights_per_step, self.out_proj.weight, x, offset_cpu)
        else:
            x = self.out_proj(x)
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x


@dataclass
class _LayerState:
    offset_cpu: int

    def reset(self):
        self.offset_cpu = 0


class StreamingTransformerLayer(StreamingModule[_LayerState]):
    """TransformerLayer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        norm (str): Normalization to use. Currently, only 'layer_norm' is supported.
        layer_scale (float, optional): If not None, LayerScale will be used with the given value as initial scale.
        gating (str): if provided, replaces FFN with special gating, like GLU, GSiGLU etc.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        skip_self_attn: If true, skips the self attention module and the norm
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        norm: str = "layer_norm",
        layer_scale: tp.Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        activation=F.gelu,
        skip_self_attn: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Redefine self_attn to our streaming multi-head attention
        attn_kwargs: tp.Dict[str, tp.Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        if not skip_self_attn:
            self.self_attn: StreamingMultiheadAttention = StreamingMultiheadAttention(
                causal=causal,
                context=context,
                rope=rope,
                weights_per_step=weights_per_step,
                **attn_kwargs,  # type: ignore
                **factory_kwargs,  # type: ignore
            )  # type: ignore
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        # Redefine feedforward layers to expose bias parameter
        self.weights_per_step = weights_per_step
        self.gating: tp.Optional[nn.Module] = None
        self.linear1: tp.Optional[nn.Module] = None
        self.linear2: tp.Optional[nn.Module] = None
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        if isinstance(dim_feedforward, list):
            assert dim_feedforward
            assert len(dim_feedforward) == weights_per_step, (
                "Length of dim_feedforward must match weights_per_step,"
                f" got {len(dim_feedforward)} != {weights_per_step}"
            )
        if gating == "none":
            assert (
                not weights_per_step
            ), "weights_per_step without gating not supported for now."
            assert not isinstance(
                dim_feedforward, list
            ), "List dim_feedforward without gating not supported for now."
            self.linear1 = nn.Linear(
                d_model, dim_feedforward, bias=False, **factory_kwargs
            )
            self.linear2 = nn.Linear(
                dim_feedforward, d_model, bias=False, **factory_kwargs
            )
        else:
            self.linear1 = None
            self.linear2 = None
            if weights_per_step:
                if isinstance(dim_feedforward, int):
                    dim_feedforward = [dim_feedforward] * weights_per_step
                assert isinstance(dim_feedforward, list), dim_feedforward
                self.gating = nn.ModuleList(
                    [
                        make_gating(gating, d_model, dim, **factory_kwargs)
                        for dim in dim_feedforward
                    ]
                )
            else:
                assert isinstance(dim_feedforward, int)
                self.gating = make_gating(
                    gating, d_model, dim_feedforward, **factory_kwargs
                )

        self.layer_scale_1: nn.Module
        self.layer_scale_2: nn.Module
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore

    def _init_streaming_state(self, batch_size: int) -> _LayerState:
        return _LayerState(offset_cpu=0)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        state = self._streaming_state
        offset = 0
        if state is not None:
            offset = state.offset_cpu
        x_orig = x
        x = self.norm2(x)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                B, T, D = x.shape
                ys = []
                for t in range(T):
                    y = self.gating[offset + t](x[:, t : t + 1])
                    ys.append(y)
                update = torch.cat(ys, dim=1)
            else:
                update = self.gating(x)
        return x_orig + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor):
        if self.skip_self_attn:
            return x
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, x, x)
        return x_orig + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor):
        with ExitStack() as stack:
            if x.device.type != 'cuda':
                stack.enter_context(no_compile())
            x = self._sa_block(x)
            x = self._ff_block(x)
            state = self._streaming_state
            if state:
                state.offset_cpu += x.shape[1]
            return x


@dataclass
class _TransformerState:
    offset: torch.Tensor

    def reset(self):
        self.offset.zero_()


class StreamingTransformer(StreamingModule[_TransformerState]):
    """Transformer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        layer_scale (float, optional): If not None, LayerScale will be used
            with the given value as initial scale.
        positional_embedding (str): Positional embedding strategy (sin, rope, sin_rope, or none).
        max_period (float): Maximum period of the time embedding.
        positional_scale (float): Scale of positional embedding, set to 0 to deactivate.
        layer_class: (subclass of `StreamingTransformerLayer): class to use
            to initialize the layers, allowing further customization outside of AudioCraft.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `StreamingTransformerLayer`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = _streaming_device(self, next(self.parameters()).device)
        return _TransformerState(offset=torch.zeros(1, device=device, dtype=torch.long))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        state = self._streaming_state
        if state is None:
            offset = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            offset = state.offset

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offset.view(-1, 1, 1)
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=x.dtype
            )
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        if state is not None:
            state.offset.add_(T)
        return x


class ProjectedTransformer(StreamingContainer):
    """Transformer with optional projections of the input and output to different dimensions when needed.
    Supports multiple outputs.

    Args:
        input_dimension (int): dimension of the input.
        output_dimensions (tuple[int]): dimensions of the outputs.
        d_model (int): inner dimension of the Transformer.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            Similarly, the output will have the same layout.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tp.Tuple[int, ...],
        d_model: int,
        *,
        conv_layout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.conv_layout = conv_layout
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(
                    nn.Linear(d_model, output_dimension, bias=False)
                )

    def forward(self, x, *args, **kwargs):
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, *args, **kwargs)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            if self.conv_layout:
                y = y.transpose(1, 2)
            ys.append(y)
        return ys
