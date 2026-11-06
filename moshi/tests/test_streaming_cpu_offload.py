"""Streaming state must not be allocated on the `meta` device under accelerate CPU offload."""

import pytest
import torch

accelerate = pytest.importorskip("accelerate")

from moshi.modules.transformer import StreamingMultiheadAttention, StreamingTransformer


def test_streaming_attention_with_cpu_offload():
    torch.manual_seed(0)
    m = StreamingMultiheadAttention(
        embed_dim=32, num_heads=4, causal=True, context=16, device="cpu", dtype=torch.float32
    )
    x = torch.randn(1, 3, 32)
    with torch.no_grad(), m.streaming(1):
        ref = m(x, x, x)
    m = accelerate.cpu_offload(m, execution_device=torch.device("cpu"))
    assert m.in_proj_weight.device.type == "meta"
    with torch.no_grad(), m.streaming(1):
        out = m(x, x, x)
    assert torch.allclose(out, ref, atol=1e-5)


def test_streaming_transformer_with_cpu_offload():
    torch.manual_seed(0)
    m = StreamingTransformer(
        d_model=32,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
        causal=True,
        context=16,
        positional_embedding="rope",
        device="cpu",
        dtype=torch.float32,
    )
    x = torch.randn(1, 3, 32)
    with torch.no_grad(), m.streaming(1):
        ref = m(x)
    m = accelerate.cpu_offload(m, execution_device=torch.device("cpu"))
    with torch.no_grad(), m.streaming(1):
        out = m(x)
    assert torch.allclose(out, ref, atol=1e-5)
