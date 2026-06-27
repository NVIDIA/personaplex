# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Convert PersonaPlex/Moshi checkpoints into an MLX runtime bundle.

The converter preserves a standard safetensors copy for validation and creates
the native ``moshi_mlx`` weight layout used by the MLX web runtime on Apple
Silicon. It also copies local side assets that are useful for offline startup.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import struct
import tempfile
from typing import Any


WEIGHTS_NAME = "model.safetensors"
MANIFEST_NAME = "mlx_manifest.json"
TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_REPO = "nvidia/personaplex-7b-v1"
MOSHI_MLX_NATIVE_NAME = "model.moshi_mlx.safetensors"
MOSHI_MLX_CONFIG_NAME = "config.json"
PERSONAPLEX_MLX_CONFIG = {
    "model_type": "personaplex",
    "version": "7b-v1",
    "moshi_name": MOSHI_MLX_NATIVE_NAME,
    "mimi_name": "tokenizer.safetensors",
    "tokenizer_name": TEXT_TOKENIZER_NAME,
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": 2048,
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
    "depformer_dim_feedforward": 4224,
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
SAFETENSORS_DTYPES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


def _atomic_copy(source: Path, destination: Path, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists. Pass --force to overwrite it.")

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        with source.open("rb") as src:
            shutil.copyfileobj(src, tmp, length=1024 * 1024 * 64)
    shutil.copystat(source, tmp_path)
    tmp_path.replace(destination)


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_size = struct.unpack("<Q", size_bytes)[0]
        header_bytes = handle.read(header_size)
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} has an invalid safetensors header") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path} has an invalid safetensors header")
    return header_size, header


def _tensor_entries(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: value
        for name, value in header.items()
        if name != "__metadata__" and isinstance(value, dict)
    }


def _shape_numel(shape: list[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _summarize_safetensors(path: Path) -> dict[str, Any]:
    header_size, header = _read_safetensors_header(path)
    tensors = _tensor_entries(header)
    if not tensors:
        raise ValueError(f"{path} does not contain any tensors")

    dtype_counts: Counter[str] = Counter()
    total_tensor_bytes = 0
    max_data_offset = 0
    samples: list[dict[str, Any]] = []

    for name in sorted(tensors):
        meta = tensors[name]
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if dtype not in SAFETENSORS_DTYPES:
            raise ValueError(f"{path}: tensor {name!r} has unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
            raise ValueError(f"{path}: tensor {name!r} has invalid shape {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
            or offsets[0] > offsets[1]
        ):
            raise ValueError(f"{path}: tensor {name!r} has invalid data offsets {offsets!r}")

        expected_bytes = _shape_numel(shape) * SAFETENSORS_DTYPES[dtype]
        actual_bytes = offsets[1] - offsets[0]
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{path}: tensor {name!r} byte size mismatch: "
                f"header says {actual_bytes}, shape/dtype imply {expected_bytes}"
            )

        dtype_counts[dtype] += 1
        total_tensor_bytes += actual_bytes
        max_data_offset = max(max_data_offset, offsets[1])
        if len(samples) < 8:
            samples.append({"name": name, "dtype": dtype, "shape": shape})

    expected_size = 8 + header_size + max_data_offset
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: file size mismatch: got {actual_size}, safetensors header implies {expected_size}"
        )

    return {
        "file": path.name,
        "file_size": actual_size,
        "tensor_count": len(tensors),
        "total_tensor_bytes": total_tensor_bytes,
        "dtypes": dict(sorted(dtype_counts.items())),
        "samples": samples,
    }


def _resolve_checkpoint(input_path: str | None, hf_repo: str, filename: str) -> Path:
    if input_path is None:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(hf_repo, filename))

    path = Path(input_path).expanduser()
    if path.is_dir():
        exact = path / filename
        if exact.exists():
            return exact
        fallback = path / WEIGHTS_NAME
        if fallback.exists():
            return fallback
        candidates = sorted(path.glob("*.safetensors"))
        candidates = [candidate for candidate in candidates if "tokenizer" not in candidate.name]
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            f"Could not find {filename!r} in {path}. Pass --input with the checkpoint file path."
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _asset_candidates(source_root: Path, checkpoint: Path) -> list[Path]:
    names = [
        "config.json",
        TEXT_TOKENIZER_NAME,
        MIMI_NAME,
        "tokenizer.safetensors",
        "voices.tgz",
        "dist.tgz",
    ]
    candidates: list[Path] = []
    for name in names:
        path = source_root / name
        if path.exists() and path.resolve() != checkpoint.resolve():
            candidates.append(path)
    voices_dir = source_root / "voices"
    if voices_dir.exists():
        candidates.append(voices_dir)
    return candidates


def _copy_assets(source_root: Path, checkpoint: Path, output_dir: Path, force: bool) -> list[str]:
    copied: list[str] = []
    if source_root == checkpoint.parent and not source_root.exists():
        return copied

    for source in _asset_candidates(source_root, checkpoint):
        destination = output_dir / source.name
        if source.is_dir():
            if destination.exists():
                if not force:
                    raise FileExistsError(f"{destination} already exists. Pass --force to overwrite it.")
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            _atomic_copy(source, destination, force=force)
        copied.append(source.name)
    return copied


def _write_moshi_mlx_config(output_dir: Path) -> str:
    config_path = output_dir / MOSHI_MLX_CONFIG_NAME
    config_path.write_text(json.dumps(PERSONAPLEX_MLX_CONFIG, indent=2, sort_keys=True) + "\n")
    return config_path.name


def _convert_to_moshi_mlx_native(source: Path, output_dir: Path, mode: str, force: bool) -> dict[str, Any]:
    if mode == "off":
        return {"status": "skipped", "reason": "disabled"}
    try:
        import mlx.core as mx
        from moshi_mlx import models
    except Exception as exc:
        if mode == "auto":
            return {"status": "skipped", "reason": f"moshi_mlx unavailable: {exc}"}
        raise RuntimeError("moshi_mlx is required for --moshi-mlx-native on") from exc

    destination = output_dir / MOSHI_MLX_NATIVE_NAME
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists. Pass --force to overwrite it.")

    config = models.LmConfig.from_config_dict(PERSONAPLEX_MLX_CONFIG)
    model = models.Lm(config)
    model.set_dtype(mx.bfloat16)
    _load_personaplex_pytorch_weights(model, str(source), mx, source_depformer_chunks=16, strict=True)

    tmp_path = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    model.save_weights(str(tmp_path))
    tmp_path.replace(destination)
    summary = _summarize_safetensors(destination)
    return {
        "status": "created",
        "file": destination.name,
        "weights": summary,
    }


def _load_personaplex_pytorch_weights(model, file: str, mx, source_depformer_chunks: int, strict: bool = True):
    pth_t = mx.load(file)
    cfg = model.cfg
    mlx_t = {
        "out_norm.weight": pth_t["out_norm.alpha"][0, 0],
    }
    for name in [
        "text_emb.out1.weight",
        "text_emb.out2.weight",
        "text_emb.weight",
        "text_linear.weight",
    ]:
        if name in pth_t:
            mlx_t[name] = pth_t[name]
    for cb_idx in range(cfg.audio_codebooks):
        mlx_t[f"audio_embs.{cb_idx}.weight"] = pth_t[f"emb.{cb_idx}.weight"]
    for key, value in sorted(pth_t.items()):
        if key.startswith("transformer"):
            if key.endswith(".alpha"):
                value = value[0, 0]
            key = key.replace(".alpha", ".weight")
            key = key.replace(".in_proj_weight", ".in_proj.weight")
            mlx_t[key] = value

    for slice_idx in range(cfg.depformer.num_slices):
        pth_idx = slice_idx
        slice_p = f"depformer.slices.{slice_idx}"
        mlx_t[f"{slice_p}.linear_in.weight"] = pth_t[f"depformer_in.{pth_idx}.weight"]
        mlx_t[f"{slice_p}.linear_out.weight"] = pth_t[f"linears.{slice_idx}.weight"]
        if slice_idx == 0:
            mlx_t[f"{slice_p}.emb.weight"] = pth_t["depformer_text_emb.weight"]
            for name in ["low_rank", "out1", "out2"]:
                source_key = f"depformer_text_emb.{name}.weight"
                if source_key in pth_t:
                    mlx_t[f"{slice_p}.emb.{name}.weight"] = pth_t[source_key]
        else:
            mlx_t[f"{slice_p}.emb.weight"] = pth_t[f"depformer_emb.{slice_idx - 1}.weight"]
            source_key = f"depformer_emb.{slice_idx - 1}.low_rank.weight"
            if source_key in pth_t:
                mlx_t[f"{slice_p}.emb.low_rank.weight"] = pth_t[source_key]

        for layer_idx in range(cfg.depformer.transformer.num_layers):
            p = f"{slice_p}.transformer.layers.{layer_idx}"
            mlx_t[f"{p}.norm1.weight"] = pth_t[f"depformer.layers.{layer_idx}.norm1.alpha"][0, 0]
            mlx_t[f"{p}.norm2.weight"] = pth_t[f"depformer.layers.{layer_idx}.norm2.alpha"][0, 0]
            mlx_t[f"{p}.gating.linear_in.weight"] = pth_t[
                f"depformer.layers.{layer_idx}.gating.{pth_idx}.linear_in.weight"
            ]
            mlx_t[f"{p}.gating.linear_out.weight"] = pth_t[
                f"depformer.layers.{layer_idx}.gating.{pth_idx}.linear_out.weight"
            ]
            mlx_t[f"{p}.self_attn.in_proj.weight"] = mx.split(
                pth_t[f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"],
                source_depformer_chunks,
            )[pth_idx]
            mlx_t[f"{p}.self_attn.out_proj.weight"] = mx.split(
                pth_t[f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"],
                source_depformer_chunks,
            )[pth_idx]
    return model.load_weights(list(mlx_t.items()), strict=strict)


def _load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            nested = state.get(key)
            if isinstance(nested, dict):
                return nested
        return state
    raise TypeError(f"{path} did not load to a state dict")


def _torch_dtype(dtype: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def _mlx_dtype(dtype: str):
    import mlx.core as mx

    return {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }[dtype]


def _convert_with_torch_and_mlx(source: Path, destination: Path, dtype: str, force: bool) -> None:
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists. Pass --force to overwrite it.")

    import torch
    import mlx.core as mx

    state = _load_state_dict(source)
    target_torch_dtype = None if dtype == "keep" else _torch_dtype(dtype)
    target_mlx_dtype = None if dtype == "keep" else _mlx_dtype(dtype)
    weights: dict[str, Any] = {}

    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        if target_torch_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(target_torch_dtype)
        if tensor.dtype == torch.bfloat16:
            array = mx.array(tensor.to(torch.float32).numpy(), dtype=mx.bfloat16)
        elif target_mlx_dtype is not None and tensor.is_floating_point():
            array = mx.array(tensor.numpy(), dtype=target_mlx_dtype)
        else:
            array = mx.array(tensor.numpy())
        weights[name] = array

    if not weights:
        raise ValueError(f"{source} did not contain tensor weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.tmp")
    mx.save_safetensors(str(tmp_path), weights, metadata={"format": "mlx"})
    tmp_path.replace(destination)


def _validate_with_mlx(path: Path, expected_summary: dict[str, Any], limit: int) -> dict[str, Any]:
    import mlx.core as mx

    weights = mx.load(str(path))
    if not isinstance(weights, dict):
        raise ValueError(f"mlx.core.load({path}) did not return a weight dict")
    if len(weights) != expected_summary["tensor_count"]:
        raise ValueError(
            f"MLX loaded {len(weights)} tensors, expected {expected_summary['tensor_count']}"
        )

    checked = 0
    for sample in expected_summary["samples"][:limit]:
        name = sample["name"]
        if name not in weights:
            raise ValueError(f"MLX did not load sample tensor {name!r}")
        shape = tuple(int(dim) for dim in sample["shape"])
        if tuple(weights[name].shape) != shape:
            raise ValueError(
                f"MLX tensor {name!r} has shape {tuple(weights[name].shape)}, expected {shape}"
            )
        checked += 1
    return {
        "backend": "mlx.core.load",
        "status": "passed",
        "tensor_count": len(weights),
        "sample_tensors_checked": checked,
    }


def convert_checkpoint(
    input_path: str | None,
    output_dir: str,
    hf_repo: str = DEFAULT_REPO,
    filename: str = WEIGHTS_NAME,
    dtype: str = "keep",
    copy_assets: bool = True,
    force: bool = False,
    skip_mlx_validation: bool = False,
    validation_limit: int = 8,
    moshi_mlx_native: str = "auto",
) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(input_path, hf_repo=hf_repo, filename=filename)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / WEIGHTS_NAME

    if checkpoint.suffix == ".safetensors" and dtype == "keep":
        _atomic_copy(checkpoint, destination, force=force)
    else:
        _convert_with_torch_and_mlx(checkpoint, destination, dtype=dtype, force=force)

    source_summary = _summarize_safetensors(checkpoint) if checkpoint.suffix == ".safetensors" else None
    output_summary = _summarize_safetensors(destination)
    if source_summary is not None:
        comparable_keys = ("tensor_count", "total_tensor_bytes", "dtypes")
        for key in comparable_keys:
            if source_summary[key] != output_summary[key]:
                raise ValueError(
                    f"Converted checkpoint summary mismatch for {key}: "
                    f"{output_summary[key]!r} != {source_summary[key]!r}"
                )

    assets = _copy_assets(
        checkpoint.parent if checkpoint.parent.exists() else Path("."),
        checkpoint,
        output,
        force=force,
    ) if copy_assets else []
    config_asset = _write_moshi_mlx_config(output)
    if config_asset not in assets:
        assets.append(config_asset)

    moshi_mlx = _convert_to_moshi_mlx_native(destination, output, mode=moshi_mlx_native, force=force)

    if skip_mlx_validation:
        mlx_validation = {"backend": "mlx.core.load", "status": "skipped"}
    else:
        mlx_validation = _validate_with_mlx(destination, output_summary, limit=validation_limit)

    manifest = {
        "format": "mlx-safetensors",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(checkpoint),
        "source_type": "safetensors" if checkpoint.suffix == ".safetensors" else "torch",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "weights": output_summary,
        "assets": assets,
        "moshi_mlx": moshi_mlx,
        "mlx_validation": mlx_validation,
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _self_test() -> None:
    import numpy as np
    from safetensors.numpy import save_file

    with tempfile.TemporaryDirectory(prefix="moshi-mlx-self-test-") as tmp:
        root = Path(tmp)
        source_dir = root / "standard"
        source_dir.mkdir()
        save_file(
            {
                "linear.bias": np.arange(4, dtype=np.float32),
                "linear.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
            },
            source_dir / WEIGHTS_NAME,
            metadata={"source": "self-test"},
        )
        (source_dir / "config.json").write_text('{"model_type":"self-test"}\n')
        output_dir = root / "mlx"
        manifest = convert_checkpoint(
            str(source_dir),
            str(output_dir),
            copy_assets=True,
            force=True,
            skip_mlx_validation=False,
            moshi_mlx_native="off",
        )
        if manifest["weights"]["tensor_count"] != 2:
            raise AssertionError("self-test tensor count mismatch")
        if not (output_dir / "config.json").exists():
            raise AssertionError("self-test asset copy failed")
        print(f"Self-test passed: {output_dir / WEIGHTS_NAME}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a standard PersonaPlex/Moshi checkpoint into an MLX runtime bundle."
    )
    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Checkpoint file or model directory. If omitted, the checkpoint is downloaded "
            "from --hf-repo using --filename."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help=f"Directory to write {WEIGHTS_NAME}, assets, and {MANIFEST_NAME}.",
    )
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--filename", type=str, default=WEIGHTS_NAME)
    parser.add_argument(
        "--dtype",
        choices=("keep", "float16", "bfloat16", "float32"),
        default="keep",
        help=(
            "Output floating dtype. 'keep' preserves safetensors byte-for-byte. "
            "Changing dtype or converting .bin/.pt requires torch and MLX."
        ),
    )
    parser.add_argument("--no-assets", action="store_true", help="Do not copy config/tokenizer/voice assets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument(
        "--skip-mlx-validation",
        action="store_true",
        help="Skip the final mlx.core.load validation step.",
    )
    parser.add_argument(
        "--moshi-mlx-native",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Create model.moshi_mlx.safetensors for moshi_mlx runtime. "
            "'auto' creates it when moshi_mlx is installed."
        ),
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=8,
        help="Number of sample tensors to shape-check after mlx.core.load.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny conversion through safetensors and MLX, then exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")

    manifest = convert_checkpoint(
        input_path=args.input,
        output_dir=args.output_dir,
        hf_repo=args.hf_repo,
        filename=args.filename,
        dtype=args.dtype,
        copy_assets=not args.no_assets,
        force=args.force,
        skip_mlx_validation=args.skip_mlx_validation,
        validation_limit=args.validation_limit,
        moshi_mlx_native=args.moshi_mlx_native,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
