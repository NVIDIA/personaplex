from __future__ import annotations

import logging
import os
import re
from typing import Callable

import torch

from .compile import set_compile_disabled, set_cuda_graph_disabled


_DEFAULT_WARN = logging.getLogger(__name__).warning
_FORCE_FAST_RUNTIME_ENV = "PERSONAPLEX_FORCE_FAST_RUNTIME"
_H20_MIN_SAFE_TORCH = (2, 5)


def _is_cuda_requested(device: str | torch.device | None) -> bool:
    if device is None:
        return True
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return torch.device(device).type == "cuda"


def _parse_torch_version(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _force_fast_runtime_enabled() -> bool:
    value = os.environ.get(_FORCE_FAST_RUNTIME_ENV, "")
    return value.lower() not in {"", "0", "false", "no", "n"}


def apply_runtime_compatibility_guard(
    device: str | torch.device | None = None,
    warn: Callable[[str], None] | None = None,
) -> bool:
    """Disable risky CUDA fast paths for known-problematic H20 runtimes."""
    if not _is_cuda_requested(device) or not torch.cuda.is_available():
        return False

    if _force_fast_runtime_enabled():
        return False

    try:
        device_names = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception:
        return False

    if not any("H20" in name.upper() for name in device_names):
        return False

    torch_version = _parse_torch_version(torch.__version__)
    if torch_version is None or torch_version >= _H20_MIN_SAFE_TORCH:
        return False

    set_compile_disabled(True)
    set_cuda_graph_disabled(True)

    if warn is None:
        warn = _DEFAULT_WARN

    warn(
        "Detected NVIDIA H20 GPU(s) with torch %s; disabling torch.compile and CUDA "
        "graphs to avoid known SIGFPE / floating-point crashes on this runtime. "
        "Set %s=1 to keep the fast path if you have already validated it."
        % (torch.__version__, _FORCE_FAST_RUNTIME_ENV)
    )
    return True
