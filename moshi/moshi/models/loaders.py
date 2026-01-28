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
"""Retrieves the pretrained models for Moshi and Mimi."""
from pathlib import Path
import logging

from safetensors.torch import load_model, load_file
import torch

logger = logging.getLogger(__name__)

from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..modules.streaming import StreamingContainer
from ..quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'nvidia/personaplex-7b-v1'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename)
    else:
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(8)
    return model


def get_moshi_lm(
    filename: str | Path | None,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays=None,
    cpu_offload: bool = False,
    multi_gpu: bool = False,
    gpus: int | None = None,
) -> LMModel:
    """Return a pretrained Moshi LM model.

    Args:
        filename: Path to model weights.
        copy_missing_weights: Whether to copy missing weights from existing layers.
        device: Target device for the model.
        dtype: Data type for model weights.
        delays: Optional custom delays configuration.
        cpu_offload: If True, offload model layers to CPU when GPU memory is
                     insufficient. Uses accelerate's device_map="auto".
        multi_gpu: If True, distribute model across all available GPUs.
                   Uses accelerate's device_map with explicit GPU memory allocation.
        gpus: Optional limit on the number of GPUs to use when multi_gpu is True.
    """
    # Copy to avoid mutating a shared/global dict
    lm_kwargs = dict(_lm_kwargs)
    lm_kwargs["dep_q"] = 16
    if delays is not None:
        lm_kwargs["delays"] = delays

    if multi_gpu and filename is not None:
        return _get_moshi_lm_multi_gpu(
            filename, copy_missing_weights, device, dtype, lm_kwargs, gpus
        )

    if cpu_offload and filename is not None:
        return _get_moshi_lm_with_offload(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )

    # Init with meta device to avoid init dummy memory
    init_device = "meta" if filename is not None else device
    model = LMModel(device=init_device, dtype=dtype, **lm_kwargs)
    if filename is None:
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    filename = str(filename)

    # Load state_dict
    if filename.endswith(".safetensors"):
        # safetensors does not support mps directly
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type == "mps":
            state_dict = load_file(filename, device="cpu")
        else:
            state_dict = load_file(filename, device=dev.type)
    else:
        # torch checkpoint
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
    # Patch 1: expand depformer self_attn weights if needed
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                print("Expanding %s", name)
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0] :]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    # Patch 2: fill missing keys by copying 0..7 -> 8..15 for certain groups
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            print("Replacing %s <- %s", name, src)
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                print("Missing %s", name)

    # Assign weights to target device
    dev = torch.device(device) if isinstance(device, str) else device
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device=dev, dtype=dtype)
    
    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    return model.to(device=device, dtype=dtype)


def _get_moshi_lm_with_offload(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM with CPU offloading using accelerate.

    This function distributes model layers across GPU and CPU based on
    available GPU memory. Layers that don't fit on GPU are kept on CPU
    and moved to GPU only during forward pass.
    """
    try:
        from accelerate import infer_auto_device_map, dispatch_model
    except ImportError:
        raise ImportError(
            "CPU offloading requires the 'accelerate' package. "
            "Install it with: pip install accelerate"
        )

    filename = str(filename)
    logger.info("Loading model with CPU offloading enabled")

    # First, create model on CPU to get the architecture
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    # Load state_dict to CPU
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device="cpu")
    else:
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

    # Apply weight patches (same as non-offload path)
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                logger.warning(f"Missing {name}")

    model.load_state_dict(state_dict, strict=False, assign=True)

    # Determine target device
    dev = torch.device(device) if isinstance(device, str) else device

    if dev.type != "cuda":
        # If not using CUDA, just move to the target device without offloading
        logger.info(f"CPU offload requested but device is {dev}, skipping offload")
        model.to(dev)
        model.eval()
        return model

    # Infer device map based on available GPU memory
    device_map = infer_auto_device_map(
        model,
        max_memory=None,  # Let accelerate auto-detect available memory
        no_split_module_classes=["StreamingTransformerLayer"],
        dtype=dtype,
    )

    # Log the device distribution
    gpu_layers = sum(1 for v in device_map.values() if v == 0 or v == "cuda:0")
    cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
    logger.info(f"Device map: {gpu_layers} modules on GPU, {cpu_layers} modules on CPU")

    # Dispatch model across devices
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir="offload_weights",  # Directory for disk offload if needed
    )

    model.eval()
    return model


def _get_moshi_lm_multi_gpu(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
    gpus: int | None = None,
) -> LMModel:
    """Load Moshi LM distributed across multiple GPUs using manual pipeline parallelism.

    Distributes transformer layers across GPUs and handles device transitions
    explicitly. This approach is compatible with Moshi's streaming KV cache.
    """
    filename = str(filename)
    available_gpus = torch.cuda.device_count()
    num_gpus = available_gpus
    if gpus is not None:
        if gpus > available_gpus:
            logger.warning(f"Requested {gpus} GPUs but only {available_gpus} are available.")
        else:
            num_gpus = gpus
    
    logger.info(f"Multi-GPU: found {available_gpus} CUDA devices, using {num_gpus}")

    if num_gpus < 2:
        raise RuntimeError(
            f"--multi-gpu requires at least 2 GPUs, but found {num_gpus}. "
            "Use --cpu-offload for single GPU with CPU fallback."
        )

    # Create model on CPU first
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    # Load weights to CPU
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device="cpu")
    else:
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

    # Apply weight patches
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                logger.warning(f"Missing {name}")

    model.load_state_dict(state_dict, strict=False, assign=True)

    # Distribute model components across GPUs
    num_transformer_layers = len(model.transformer.layers)
    layers_per_gpu = (num_transformer_layers + num_gpus - 1) // num_gpus

    logger.info(f"Distributing {num_transformer_layers} transformer layers across {num_gpus} GPUs")
    logger.info(f"Layers per GPU: ~{layers_per_gpu}")

    # Create device assignments for transformer layers
    layer_devices = []
    for i in range(num_transformer_layers):
        gpu_idx = min(i // layers_per_gpu, num_gpus - 1)
        layer_devices.append(torch.device(f"cuda:{gpu_idx}"))

    # Move transformer layers to their assigned GPUs
    for i, layer in enumerate(model.transformer.layers):
        layer.to(layer_devices[i])
        logger.debug(f"Transformer layer {i} -> {layer_devices[i]}")

    # Move embeddings and first-stage components to cuda:0
    primary_device = torch.device("cuda:0")
    model.emb.to(primary_device)
    model.text_emb.to(primary_device)

    # Move output components to the last GPU (where transformer output lands)
    last_device = layer_devices[-1]
    model.out_norm.to(last_device)
    model.text_linear.to(last_device)

    # Move depformer components to last GPU (where transformer output lands)
    # This eliminates the expensive transformer_out transfer back to cuda:0
    model.depformer_in.to(last_device)
    model.depformer.to(last_device)
    model.depformer_emb.to(last_device)
    model.depformer_text_emb.to(last_device)
    model.linears.to(last_device)

    # Log final distribution
    gpu_layers = {}
    for i, dev in enumerate(layer_devices):
        dev_str = str(dev)
        gpu_layers[dev_str] = gpu_layers.get(dev_str, 0) + 1
    logger.info(f"Layer distribution: {gpu_layers}")

    # Create wrapper that handles device transitions
    class MultiGPULMModel(StreamingContainer):
        """Wrapper that handles device transitions for multi-GPU inference.

        Multi-GPU Synchronization Strategy: CUDA Events
        ================================================

        Problem:
            When distributing transformer layers across multiple GPUs, data must be
            transferred between devices at layer boundaries. The naive approach of
            using torch.cuda.synchronize() after each transfer causes 18-90ms of
            CPU blocking per frame, exceeding the 80ms real-time budget.

            Simply removing synchronization (non_blocking=True only) creates data
            races: GPU 1 may start computing before GPU 0's transfer completes,
            since cross-device transfers use separate CUDA streams with no implicit
            ordering.

        Solution:
            CUDA Events provide non-blocking GPU-to-GPU synchronization:

            1. When GPU A finishes its layers, record an event on GPU A's stream
            2. Before GPU B starts, make GPU B's stream wait for GPU A's event
            3. The CPU never blocks - only the destination GPU waits if needed

            Data flow with events:
            ```
            GPU 0: [Embedding] -> [Layers 0-15] -> record(event_0)
                                                          |
                                                          v (event wait, non-blocking to CPU)
            GPU N:                                  wait(event_0) -> [Layers 16-31] -> [Depformer]
                                                                                              |
                                                                                              v
            GPU 0:                                                          output tokens (~64 bytes)
            ```

            Note: Depformer runs on last GPU (GPU N) to avoid expensive transformer_out
            transfer (~16KB). Only the small output tokens are transferred back to GPU 0.

        Performance:
            | Approach                    | CPU Blocking | Overhead/Frame |
            |-----------------------------|--------------|----------------|
            | torch.cuda.synchronize()    | Yes          | 18-90ms        |
            | non_blocking only (broken)  | No           | DATA RACES     |
            | CUDA Events (this impl)     | No           | ~4-8μs         |

        Implementation Details:
            - _gpu_streams: One dedicated CUDA stream per GPU for layer execution
            - _boundary_events: One CUDA event per GPU to signal completion
            - forward_embeddings(): Records events at device boundaries, waits before
              starting on new device, executes layers within stream contexts
            - Final transfer back to primary device also uses event synchronization
            - wait_stream() syncs with default stream before depformer execution

        References:
            - PyTorch CUDA Semantics: https://pytorch.org/docs/stable/notes/cuda.html
            - CUDA Events: https://pytorch.org/docs/stable/generated/torch.cuda.Event.html
        """

        def __init__(self, model, layer_devices, primary_device, last_device):
            super().__init__()
            # Register as a named module so streaming state propagates to underlying model
            self.add_module('_model', model)
            self._layer_devices = layer_devices
            self._primary_device = primary_device
            self._last_device = last_device
            self._depformer_device = last_device  # Depformer runs on last GPU
            self._is_multi_gpu = True  # Flag for CUDA graph detection
            # Track device boundaries for efficient transitions
            self._device_boundaries = []
            current_device = layer_devices[0]
            for i, dev in enumerate(layer_devices[1:], 1):
                if dev != current_device:
                    self._device_boundaries.append(i)
                    current_device = dev

            # Create per-GPU streams and events for synchronization
            # CUDA events allow non-blocking synchronization between GPUs
            self._gpu_streams = {}
            self._boundary_events = {}

            for dev in set(layer_devices):
                dev_str = str(dev)
                if dev_str not in self._gpu_streams:
                    self._gpu_streams[dev_str] = torch.cuda.Stream(device=dev)
                    self._boundary_events[dev_str] = torch.cuda.Event()

            # Also for primary device
            primary_str = str(primary_device)
            if primary_str not in self._gpu_streams:
                self._gpu_streams[primary_str] = torch.cuda.Stream(device=primary_device)
                self._boundary_events[primary_str] = torch.cuda.Event()

            # Per-device CUDA graphs for layer batches
            # Each GPU's contiguous layer range can be graphed independently
            self._layer_graphs = {}
            self._layer_graph_inputs = {}
            self._layer_graph_outputs = {}
            self._layer_warmup_remaining = {}
            for dev_str in self._gpu_streams.keys():
                self._layer_graphs[dev_str] = None  # Will be created on first use
                self._layer_graph_inputs[dev_str] = None
                self._layer_graph_outputs[dev_str] = None
                self._layer_warmup_remaining[dev_str] = 2  # Warmup iterations

            # Precompute layer ranges per device for graphing
            self._device_layer_ranges = {}
            current_dev = None
            range_start = 0
            for i, dev in enumerate(layer_devices):
                dev_str = str(dev)
                if dev_str != current_dev:
                    if current_dev is not None:
                        if current_dev not in self._device_layer_ranges:
                            self._device_layer_ranges[current_dev] = []
                        self._device_layer_ranges[current_dev].append((range_start, i))
                    current_dev = dev_str
                    range_start = i
            # Don't forget the last range
            if current_dev is not None:
                if current_dev not in self._device_layer_ranges:
                    self._device_layer_ranges[current_dev] = []
                self._device_layer_ranges[current_dev].append((range_start, len(layer_devices)))

        # HISTORY: Originally used torch.cuda.synchronize() at device boundaries,
        # which caused 18-90ms CPU blocking per frame. A "quick fix" removed all
        # synchronization, but this created data races since cross-device transfers
        # use separate CUDA streams with no implicit ordering.
        #
        # CURRENT: Uses CUDA events (see _boundary_events) for non-blocking GPU-to-GPU
        # synchronization. The destination GPU waits for the source GPU's event,
        # but the CPU never blocks. Overhead is ~4-8μs per frame.

        @property
        def device(self):
            return self._primary_device

        def forward(self, *args, **kwargs):
            return self._model(*args, **kwargs)

        def embed_codes(self, sequence):
            # Embeddings are on primary device
            sequence = sequence.to(self._primary_device, non_blocking=True)
            return self._model.embed_codes(sequence)

        def forward_embeddings(self, input_):
            """Forward pass through transformer with CUDA event synchronization.

            This method processes input through all transformer layers distributed
            across multiple GPUs, using CUDA events for non-blocking synchronization
            at device boundaries.

            Synchronization Flow:
                1. Each GPU has a dedicated stream for layer execution
                2. At device boundaries (GPU A -> GPU B):
                   a. Record completion event on GPU A's stream
                   b. Switch to GPU B's stream
                   c. Wait for GPU A's event (GPU-side wait, CPU doesn't block)
                   d. Transfer tensor to GPU B (non_blocking=True)
                3. After all layers, transfer output back to primary device
                4. Sync with default stream so depformer sees the data

            Args:
                input_: Embedded input tensor, shape (B, T, C)

            Returns:
                Tuple of (transformer_output, text_logits), both on primary device
            """
            from ..modules.transformer import create_sin_embedding

            model = self._model
            transformer = model.transformer

            # Input should be on primary device (where first layers are)
            x = input_.to(self._layer_devices[0])
            B, T, C = x.shape

            # Streaming state is now properly initialized via StreamingContainer inheritance
            state = transformer._streaming_state
            if state is None:
                raise RuntimeError("Transformer streaming state not initialized. Call streaming_forever() first.")
            offset = state.offset

            # Handle positional embeddings (for "sin" or "sin_rope" modes)
            if transformer.positional_embedding in {"sin", "sin_rope"}:
                positions = torch.arange(T, device=x.device).view(1, -1, 1)
                positions = positions + offset.view(-1, 1, 1)
                pos_emb = create_sin_embedding(
                    positions, C, max_period=transformer.max_period, dtype=x.dtype
                )
                x = x + transformer.positional_scale * pos_emb

            # Process through transformer layers with device transitions
            # Using CUDA events for non-blocking synchronization between GPUs
            current_device = self._layer_devices[0]
            current_stream = self._gpu_streams[str(current_device)]

            # Ensure the custom stream waits for any pending operations on the default stream
            # This is critical because:
            # - embed_codes() uses non_blocking=True for the sequence transfer
            # - Input transfer at line 635 happens on the default stream
            # Without this sync, the first layer might execute before data is ready
            current_stream.wait_stream(torch.cuda.current_stream(current_device))

            for i, layer in enumerate(transformer.layers):
                layer_device = self._layer_devices[i]

                if layer_device != current_device:
                    # Record event when current GPU finishes
                    with torch.cuda.stream(current_stream):
                        self._boundary_events[str(current_device)].record()

                    # Switch to new device's stream
                    prev_device = current_device
                    current_device = layer_device
                    current_stream = self._gpu_streams[str(current_device)]

                    # Wait for previous GPU before starting
                    with torch.cuda.stream(current_stream):
                        self._boundary_events[str(prev_device)].wait()
                        x = x.to(layer_device, non_blocking=True)

                # Execute layer on current stream
                with torch.cuda.stream(current_stream):
                    x = layer(x)

            # Record final event
            with torch.cuda.stream(current_stream):
                self._boundary_events[str(current_device)].record()

            # Update streaming state offset
            if state is not None:
                state.offset.add_(T)

            # Output norm and text linear on last device's stream
            last_stream = self._gpu_streams[str(self._last_device)]
            with torch.cuda.stream(last_stream):
                if model.out_norm:
                    x = model.out_norm(x)
                text_logits = model.text_linear(x)
                text_logits = text_logits[:, None]
                self._boundary_events[str(self._last_device)].record()

            # Transfer text_logits back to primary device for sampling
            # (transformer_out stays on last_device for depformer)
            primary_stream = self._gpu_streams[str(self._primary_device)]
            with torch.cuda.stream(primary_stream):
                self._boundary_events[str(self._last_device)].wait()
                text_logits = text_logits.to(self._primary_device, non_blocking=True)

            # Sync streams: last_device for depformer (uses x), primary for sampling (uses text_logits)
            torch.cuda.current_stream(self._last_device).wait_stream(last_stream)
            torch.cuda.current_stream(self._primary_device).wait_stream(primary_stream)

            return x, text_logits

        def forward_codes(self, sequence):
            """Forward pass from codes through embeddings and transformer."""
            input_ = self.embed_codes(sequence)
            return self.forward_embeddings(input_)

        def forward_depformer(self, *args, skip_transfer: bool = False, **kwargs):
            """Forward pass through depformer on last device.

            No explicit synchronization needed here because forward_embeddings()
            already synchronized with the default stream via wait_stream() before
            returning. The depformer inputs are guaranteed to be ready.

            Args:
                skip_transfer: If True, skip device transfers for inputs (they are
                    already on depformer device). Used by CUDA-graphed depformer_step
                    where transfers happen outside the graph capture.

            Transfers input tensors to depformer device if needed (unless skip_transfer),
            and returns result without transferring back (caller handles that).
            """
            if not skip_transfer:
                # Transfer inputs to depformer device
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.device != self._depformer_device:
                        arg = arg.to(self._depformer_device, non_blocking=True)
                    new_args.append(arg)
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.device != self._depformer_device:
                        v = v.to(self._depformer_device, non_blocking=True)
                    new_kwargs[k] = v
            else:
                # Inputs already on depformer device, skip transfer
                new_args = list(args)
                new_kwargs = dict(kwargs)

            # Pass skip_transfer to underlying model (no-op for base LMModel but keeps signature consistent)
            new_kwargs['skip_transfer'] = skip_transfer
            result = self._model.forward_depformer(*new_args, **new_kwargs)

            # When skip_transfer=True, caller handles result transfer
            # When skip_transfer=False (legacy path), transfer result back to primary device
            if not skip_transfer:
                if isinstance(result, torch.Tensor):
                    result = result.to(self._primary_device, non_blocking=True)
                elif isinstance(result, tuple):
                    result = tuple(
                        r.to(self._primary_device, non_blocking=True) if isinstance(r, torch.Tensor) else r
                        for r in result
                    )
            return result

        def _get_initial_token(self):
            return self._model._get_initial_token().to(self._primary_device)

        def parameters(self, recurse=True):
            # Yield embeddings first (on primary device) to ensure device detection works
            # This ensures next(iter(model.parameters())).device returns cuda:0
            yield from self._model.emb.parameters(recurse)
            yield from self._model.text_emb.parameters(recurse)
            # Then other parameters
            for name, param in self._model.named_parameters(recurse=recurse):
                if not name.startswith('emb.') and not name.startswith('text_emb.'):
                    yield param

        def __getattr__(self, name):
            if name.startswith('_') or name in ('training', 'forward', 'embed_codes',
                                                  'forward_embeddings', 'forward_codes',
                                                  'forward_depformer', 'device', '_depformer_device'):
                return super().__getattr__(name)
            return getattr(self._model, name)

    wrapped = MultiGPULMModel(model, layer_devices, primary_device, last_device)
    wrapped.eval()

    # Log memory usage
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"GPU {i} memory allocated: {allocated:.2f} GB")

    return wrapped
