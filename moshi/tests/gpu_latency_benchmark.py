"""Measure active/shadow PersonaPlex latency on two resident GPUs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
from typing import Any
import wave

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from moshi.models import loaders
from moshi.seamless.controller import SeamlessInstallController
from moshi.seamless.memory import NullMemoryWorker
from moshi.seamless.model_slot import (
    AsyncPersonaPlexSlot,
    PersonaPlexSlotEngine,
    wrap_with_system_tags,
)
from moshi.seamless.types import ControllerConfig, DuplexFrame


FRAME_RATE = 12.5
FRAME_BUDGET_MS = 1000 / FRAME_RATE


def latency_summary(values_ms: list[float]) -> dict[str, float | int | None]:
    values = np.asarray(values_ms, dtype=np.float64)
    if not values.size:
        return {
            "n": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "deadline_miss_rate": None,
        }
    return {
        "n": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "deadline_miss_rate": float(np.mean(values > FRAME_BUDGET_MS)),
    }


def memory_for_target_tokens(
    engine: PersonaPlexSlotEngine, target: int
) -> tuple[str, int]:
    if target == 0:
        return "", 0
    lines: list[str] = []
    tokens = 0
    index = 0
    while tokens < target:
        lines.append(
            f"Entity {index:03d} | preference = Aurora {index:03d} | status = current"
        )
        index += 1
        text = "\n".join(lines)
        tokens = len(engine.tokenizer.encode(wrap_with_system_tags(text)))
    return text, tokens


def mark_played(frames: list[DuplexFrame], result: Any) -> None:
    if (
        result.emitted_seq is not None
        and result.emitted_seq >= 0
        and result.output_pcm is not None
    ):
        emitted = frames[result.emitted_seq]
        emitted.played = True
        emitted.played_pcm = result.output_pcm.copy()


def load_pcm(path: Path, expected_sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise ValueError("benchmark input must be mono 16-bit PCM WAV")
        if reader.getframerate() != expected_sample_rate:
            raise ValueError(
                f"benchmark input is {reader.getframerate()} Hz, "
                f"expected {expected_sample_rate} Hz"
            )
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def audio_frame(pcm: np.ndarray, seq: int, frame_size: int) -> np.ndarray:
    start = (seq * frame_size) % pcm.size
    indices = (np.arange(frame_size) + start) % pcm.size
    return pcm[indices].copy()


async def timed_start(slot: AsyncPersonaPlexSlot, prompt: str, voice: str) -> float:
    started = time.perf_counter()
    await slot.start_session(prompt, voice)
    return time.perf_counter() - started


async def paced_active_until_done(
    slot: AsyncPersonaPlexSlot,
    background: asyncio.Task[Any] | None,
    *,
    start_seq: int,
    minimum_frames: int,
    rng: np.random.Generator,
    frame_size: int,
) -> tuple[list[float], int]:
    latencies: list[float] = []
    seq = start_seq
    deadline = time.perf_counter()
    while len(latencies) < minimum_frames or (
        background is not None and not background.done()
    ):
        pcm = (rng.standard_normal(frame_size) * 0.002).astype(np.float32)
        started = time.perf_counter()
        await slot.active_step(seq, pcm)
        latencies.append((time.perf_counter() - started) * 1000)
        seq += 1
        deadline += 1 / FRAME_RATE
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
    if background is not None:
        await background
    return latencies, seq


async def run_benchmark(
    args: argparse.Namespace,
    active_engine: PersonaPlexSlotEngine,
    shadow_engine: PersonaPlexSlotEngine,
    voice: str,
    load_timings: dict[str, float],
    input_pcm: np.ndarray,
) -> dict[str, Any]:
    active = AsyncPersonaPlexSlot(active_engine)
    shadow = AsyncPersonaPlexSlot(shadow_engine)
    rng = np.random.default_rng(args.seed)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "two-resident-GPU PersonaPlex seamless-install latency sweep",
        "frame_rate": FRAME_RATE,
        "frame_budget_ms": FRAME_BUDGET_MS,
        "devices": [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(2)
        ],
        "load": load_timings,
        "input_audio": str(args.input_audio),
    }
    try:
        started = time.perf_counter()
        warmup_times: dict[str, float] = {}

        async def warm(slot: AsyncPersonaPlexSlot, key: str) -> None:
            begin = time.perf_counter()
            await slot.warmup()
            warmup_times[key] = time.perf_counter() - begin

        # CUDA graph capture is process-global. Capturing the two resident
        # instances concurrently can invalidate both captures even though the
        # models live on different devices, so startup deliberately serializes
        # this one-time operation.
        await warm(active, "active_s")
        await warm(shadow, "shadow_s")
        report["warmup"] = {
            **warmup_times,
            "sequential_wall_s": time.perf_counter() - started,
        }
        print(json.dumps({"event": "warmup_complete", **report["warmup"]}), flush=True)

        role = "You are a helpful customer-service agent. Continue naturally."
        report["active_session_start_s"] = await timed_start(active, role, voice)

        frames: list[DuplexFrame] = []
        active_latencies: list[float] = []
        for seq in range(args.trace_frames):
            pcm = audio_frame(input_pcm, seq, active_engine.frame_size)
            begin = time.perf_counter()
            result = await active.active_step(seq, pcm)
            active_latencies.append((time.perf_counter() - begin) * 1000)
            frames.append(result.frame)
            mark_played(frames, result)
        report["active_unpaced"] = latency_summary(
            active_latencies[args.discard_frames :]
        )
        print(
            json.dumps({"event": "trace_complete", **report["active_unpaced"]}),
            flush=True,
        )

        prefill_rows: list[dict[str, Any]] = []
        memory_by_size: dict[int, str] = {}
        for target in args.memory_tokens:
            memory, actual_tokens = memory_for_target_tokens(shadow_engine, target)
            memory_by_size[target] = memory
            prompt = (
                role
                if not memory
                else f"{role}\n\n<conversation_memory>\n{memory}\n</conversation_memory>"
            )
            wall_s = await timed_start(shadow, prompt, voice)
            prefill_rows.append(
                {
                    "target_memory_tokens": target,
                    "actual_memory_tokens": actual_tokens,
                    "prompt_tokens": len(
                        shadow_engine.tokenizer.encode(wrap_with_system_tags(prompt))
                    ),
                    "wall_s": wall_s,
                }
            )
            print(json.dumps({"event": "prefill", **prefill_rows[-1]}), flush=True)
        report["prefill_sweep"] = prefill_rows

        replay_rows: list[dict[str, Any]] = []
        medium_target = args.memory_tokens[len(args.memory_tokens) // 2]
        medium_memory = memory_by_size[medium_target]
        replay_prompt = (
            f"{role}\n\n<conversation_memory>\n{medium_memory}\n</conversation_memory>"
        )
        for tail_frames in args.replay_frames:
            if tail_frames > len(frames):
                raise ValueError("replay frame sweep exceeds trace length")
            await shadow.start_session(replay_prompt, voice)
            tail = frames[-tail_frames:]
            begin = time.perf_counter()
            replay = await shadow.replay(tail)
            wall_s = time.perf_counter() - begin
            audio_s = tail_frames / FRAME_RATE
            fps = tail_frames / wall_s
            replay_rows.append(
                {
                    "frames": tail_frames,
                    "audio_s": audio_s,
                    "wall_s": wall_s,
                    "rtf": wall_s / audio_s,
                    "frames_per_s": fps,
                    "net_catchup_frames_per_s": fps - FRAME_RATE,
                    "user_code_mismatches": replay.user_code_mismatches,
                    "user_code_values_differ": replay.user_code_values_differ,
                    "user_codebook_mismatches": replay.user_codebook_mismatches,
                    "last_user_mismatch_seq": replay.last_user_mismatch_seq,
                    "target_mismatches": replay.target_mismatches,
                    "decoder_frames_compared": replay.decoder_frames_compared,
                    "decoder_mismatches": replay.decoder_mismatches,
                    "last_decoder_mismatch_seq": replay.last_decoder_mismatch_seq,
                    "last_decoder_compared_seq": replay.last_decoder_compared_seq,
                }
            )
            print(json.dumps({"event": "replay", **replay_rows[-1]}), flush=True)
        report["replay_sweep"] = replay_rows

        paced_baseline, next_seq = await paced_active_until_done(
            active,
            None,
            start_seq=args.trace_frames,
            minimum_frames=args.paced_frames,
            rng=rng,
            frame_size=active_engine.frame_size,
        )
        report["active_paced_baseline"] = latency_summary(paced_baseline)

        long_memory = memory_by_size[max(args.memory_tokens)]
        long_prompt = (
            f"{role}\n\n<conversation_memory>\n{long_memory}\n</conversation_memory>"
        )
        prefill_task = asyncio.create_task(shadow.start_session(long_prompt, voice))
        concurrent_prefill, next_seq = await paced_active_until_done(
            active,
            prefill_task,
            start_seq=next_seq,
            minimum_frames=args.paced_frames,
            rng=rng,
            frame_size=active_engine.frame_size,
        )
        report["active_during_shadow_prefill"] = latency_summary(concurrent_prefill)

        await shadow.start_session(replay_prompt, voice)
        replay_task = asyncio.create_task(
            shadow.replay(frames[-max(args.replay_frames) :])
        )
        concurrent_replay, next_seq = await paced_active_until_done(
            active,
            replay_task,
            start_seq=next_seq,
            minimum_frames=args.paced_frames,
            rng=rng,
            frame_size=active_engine.frame_size,
        )
        report["active_during_shadow_replay"] = latency_summary(concurrent_replay)
        del next_seq

        config = ControllerConfig(
            context_capacity_frames=3000,
            trigger_reserve_frames=1400,
            replay_batch_frames=args.replay_batch_frames,
            switch_max_lag_frames=2,
            safe_silence_frames=0,
            force_switch_after_s=0,
            install_timeout_s=180,
            ledger_max_frames=5000,
            codec_convergence_frames=16,
            decoder_convergence_frames=16,
            auto_trigger_enabled=False,
        )
        controller = SeamlessInstallController(
            (active, shadow), NullMemoryWorker(), config=config
        )
        await controller.start_session(
            session_id="latency-benchmark", role_prompt=role, voice_prompt=voice
        )
        controller_latencies: list[dict[str, Any]] = []
        seq = 0
        deadline = time.perf_counter()
        requested = False
        max_frames = int(args.controller_timeout_s * FRAME_RATE)
        while seq < max_frames:
            pcm = audio_frame(input_pcm, seq, active_engine.frame_size)
            phase_before = controller.phase.value
            begin = time.perf_counter()
            await controller.process_pcm(pcm)
            latency_ms = (time.perf_counter() - begin) * 1000
            controller_latencies.append(
                {"seq": seq, "phase": phase_before, "latency_ms": latency_ms}
            )
            seq += 1
            if seq == args.install_at_frame:
                requested = await controller.request_install(memory=medium_memory)
            if requested and controller.status()["memory_version"] == 1:
                break
            deadline += 1 / FRAME_RATE
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        status = controller.status()
        before = [
            row["latency_ms"]
            for row in controller_latencies
            if row["seq"] < args.install_at_frame
        ]
        during = [
            row["latency_ms"]
            for row in controller_latencies
            if row["seq"] >= args.install_at_frame
        ]
        report["controller_end_to_end"] = {
            "completed": status["memory_version"] == 1,
            "install_at_frame": args.install_at_frame,
            "swap_at_frame": seq - 1 if status["memory_version"] == 1 else None,
            "frames_during_install": seq - args.install_at_frame,
            "active_before_install": latency_summary(before),
            "active_during_install": latency_summary(during),
            "swap": status["last_swap"],
            "terminal_status": status,
        }
        await controller.close_session()

        report["gpu_memory"] = {
            str(index): {
                "max_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
                "max_reserved_gib": torch.cuda.max_memory_reserved(index) / 2**30,
            }
            for index in range(2)
        }
        return report
    finally:
        active.close()
        shadow.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--voice", default="NATF2.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input-audio", type=Path, default=Path("assets/test/input_service.wav")
    )
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--trace-frames", type=int, default=256)
    parser.add_argument("--discard-frames", type=int, default=16)
    parser.add_argument("--paced-frames", type=int, default=48)
    parser.add_argument("--memory-tokens", type=int, nargs="+", default=[0, 128, 384])
    parser.add_argument("--replay-frames", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--replay-batch-frames", type=int, default=64)
    parser.add_argument("--install-at-frame", type=int, default=48)
    parser.add_argument("--controller-timeout-s", type=float, default=90)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("gpu_latency_benchmark requires two visible GPUs")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    mimi = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    moshi = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    archive = Path(hf_hub_download(args.hf_repo, "voices.tgz"))
    voice = archive.parent / "voices" / args.voice
    if not voice.is_file():
        raise FileNotFoundError(voice)

    load_timings: dict[str, float] = {}
    begin = time.perf_counter()
    active_engine = PersonaPlexSlotEngine(
        name="slot-a",
        device="cuda:0",
        mimi_weight=mimi,
        moshi_weight=moshi,
        tokenizer_path=tokenizer,
    )
    load_timings["active_s"] = time.perf_counter() - begin
    print(json.dumps({"event": "active_loaded", **load_timings}), flush=True)
    begin = time.perf_counter()
    shadow_engine = PersonaPlexSlotEngine(
        name="slot-b",
        device="cuda:1",
        mimi_weight=mimi,
        moshi_weight=moshi,
        tokenizer_path=tokenizer,
    )
    load_timings["shadow_s"] = time.perf_counter() - begin
    load_timings["total_s"] = load_timings["active_s"] + load_timings["shadow_s"]
    print(json.dumps({"event": "shadow_loaded", **load_timings}), flush=True)

    input_pcm = load_pcm(args.input_audio, active_engine.mimi.sample_rate)

    report = asyncio.run(
        run_benchmark(
            args,
            active_engine,
            shadow_engine,
            str(voice),
            load_timings,
            input_pcm,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"event": "benchmark_complete", "output": str(args.output)}),
        flush=True,
    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
