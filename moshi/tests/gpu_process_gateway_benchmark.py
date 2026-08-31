"""Measure the production process-isolated seamless gateway on two GPUs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import wave

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from moshi.models import loaders
from moshi.seamless.controller import SeamlessInstallController
from moshi.seamless.memory import NullMemoryWorker
from moshi.seamless.model_slot import ProcessPersonaPlexSlot
from moshi.seamless.types import ControllerConfig


FRAME_RATE = 12.5
FRAME_BUDGET_MS = 80.0


def summary(values_ms: list[float]) -> dict[str, float | int]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "deadline_miss_rate": float(np.mean(values > FRAME_BUDGET_MS)),
    }


def load_pcm(path: Path, expected_sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != expected_sample_rate
        ):
            raise ValueError("input must be mono 16-bit PCM at the model sample rate")
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def audio_frame(pcm: np.ndarray, seq: int, frame_size: int) -> np.ndarray:
    start = (seq * frame_size) % pcm.size
    indices = (np.arange(frame_size) + start) % pcm.size
    return pcm[indices].copy()


async def run(args: argparse.Namespace) -> dict[str, object]:
    mimi = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    moshi = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    archive = Path(hf_hub_download(args.hf_repo, "voices.tgz"))
    voice = str(archive.parent / "voices" / args.voice)
    slots = tuple(
        ProcessPersonaPlexSlot(
            name=f"slot-{index}",
            device=f"cuda:{index}",
            mimi_weight=mimi,
            moshi_weight=moshi,
            tokenizer_path=tokenizer,
            process_rank=index,
            process_world_size=2,
        )
        for index in range(2)
    )
    startup_started = time.perf_counter()
    try:
        await asyncio.gather(*(slot.warmup() for slot in slots))
        startup_s = time.perf_counter() - startup_started
        pcm = load_pcm(args.input_audio, slots[0].sample_rate)
        config = ControllerConfig(
            context_capacity_frames=3000,
            trigger_reserve_frames=256,
            replay_batch_frames=64,
            switch_max_lag_frames=2,
            safe_silence_frames=0,
            force_switch_after_s=0,
            install_timeout_s=60,
            ledger_max_frames=5000,
            codec_convergence_frames=16,
            decoder_convergence_frames=16,
            auto_trigger_enabled=False,
        )
        controller = SeamlessInstallController(
            slots,
            NullMemoryWorker(),
            config=config,
        )
        role = "You are a helpful service agent."
        session_started = time.perf_counter()
        await controller.start_session(
            session_id="process-gateway-benchmark",
            role_prompt=role,
            voice_prompt=voice,
        )
        session_start_s = time.perf_counter() - session_started
        rows: list[dict[str, float | int | str]] = []
        deadline = time.perf_counter()
        requested = False
        max_frames = int(args.timeout_s * FRAME_RATE)
        for seq in range(max_frames):
            phase = controller.phase.value
            started = time.perf_counter()
            await controller.process_pcm(audio_frame(pcm, seq, slots[0].frame_size))
            rows.append(
                {
                    "seq": seq,
                    "phase": phase,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            if seq + 1 == args.install_at_frame:
                requested = await controller.request_install(
                    memory="Customer preference | project = Aurora | status = current"
                )
            if requested and controller.status()["memory_version"] == 1:
                break
            deadline += 1 / FRAME_RATE
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        status = controller.status()
        events = controller.drain_events()
        before = [
            float(row["latency_ms"])
            for row in rows
            if int(row["seq"]) < args.install_at_frame
        ]
        during = [
            float(row["latency_ms"])
            for row in rows
            if int(row["seq"]) >= args.install_at_frame
        ]
        await controller.close_session()
        return {
            "schema_version": 1,
            "method": "production process-isolated controller RPC",
            "devices": [slot.device_name for slot in slots],
            "startup_s": startup_s,
            "session_start_s": session_start_s,
            "assigned_cpus": [slot.assigned_cpus for slot in slots],
            "completed": status["memory_version"] == 1,
            "frames": len(rows),
            "baseline": summary(before),
            "during_install": summary(during),
            "events": events,
            "terminal_status": status,
        }
    finally:
        for slot in slots:
            slot.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--voice", default="NATF2.pt")
    parser.add_argument(
        "--input-audio", type=Path, default=Path("assets/test/input_service.wav")
    )
    parser.add_argument("--install-at-frame", type=int, default=48)
    parser.add_argument("--timeout-s", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two visible GPUs are required")
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
