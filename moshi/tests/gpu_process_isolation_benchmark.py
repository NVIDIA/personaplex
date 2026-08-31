"""Compare two-GPU PersonaPlex concurrency with one process per GPU."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any
import wave

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from moshi.models import loaders
from moshi.seamless.model_slot import PersonaPlexSlotEngine, partition_process_cpus
from moshi.seamless.types import DuplexFrame


FRAME_RATE = 12.5
FRAME_BUDGET_MS = 80.0


def latency_summary(values_ms: list[float]) -> dict[str, float | int]:
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


def load_pcm(path: str, expected_sample_rate: int) -> np.ndarray:
    with wave.open(path, "rb") as reader:
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


def make_engine(device: str, name: str, model: dict[str, str]) -> PersonaPlexSlotEngine:
    torch.cuda.set_device(torch.device(device))
    return PersonaPlexSlotEngine(
        name=name,
        device=device,
        mimi_weight=model["mimi"],
        moshi_weight=model["moshi"],
        tokenizer_path=model["tokenizer"],
    )


def pin_worker(rank: int) -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    assigned = partition_process_cpus(allowed, rank=rank, world_size=2)
    os.sched_setaffinity(0, assigned)
    # PersonaPlex streaming is dominated by GPU launches. Large independent
    # intra-op pools oversubscribe the Slurm cpuset and starve the replay host
    # thread when two model processes run together.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return assigned


def active_worker(
    model: dict[str, str],
    voice: str,
    input_audio: str,
    frames: int,
    model_ready: mp.synchronize.Event,
    all_models_ready: mp.synchronize.Event,
    baseline_done: mp.synchronize.Event,
    trace_queue: mp.Queue,
    trace_ready: mp.synchronize.Event,
    concurrent_start: mp.synchronize.Event,
    shadow_done: mp.synchronize.Event,
    output: mp.Queue,
) -> None:
    assigned_cpus = pin_worker(0)
    load_started = time.perf_counter()
    engine = make_engine("cuda:0", "active-process", model)
    load_s = time.perf_counter() - load_started
    warm_started = time.perf_counter()
    engine.warmup()
    warmup_s = time.perf_counter() - warm_started
    engine.start_session("You are a helpful service agent.", voice)
    pcm = load_pcm(input_audio, engine.mimi.sample_rate)
    model_ready.set()
    all_models_ready.wait()

    baseline: list[float] = []
    deadline = time.perf_counter()
    for seq in range(64):
        started = time.perf_counter()
        engine.active_step(seq, audio_frame(pcm, seq, engine.frame_size))
        baseline.append((time.perf_counter() - started) * 1000)
        deadline += 1 / FRAME_RATE
        time.sleep(max(0.0, deadline - time.perf_counter()))

    # Produce the exact duplex ledger on GPU 0, then send it to the shadow
    # process. This tests cross-GPU codec convergence, not merely replaying a
    # trace produced by the same model instance.
    engine.start_session("You are a helpful service agent.", voice)
    trace: list[DuplexFrame] = []
    for seq in range(frames):
        result = engine.active_step(seq, audio_frame(pcm, seq, engine.frame_size))
        trace.append(result.frame)
        if result.emitted_seq is not None and result.output_pcm is not None:
            emitted = trace[result.emitted_seq]
            emitted.played = True
            emitted.played_pcm = result.output_pcm.copy()
    trace_queue.put(trace)
    engine.start_session("You are a helpful service agent.", voice)
    baseline_done.set()
    trace_ready.wait()
    concurrent_start.wait()

    concurrent: list[float] = []
    seq = 0
    deadline = time.perf_counter()
    while len(concurrent) < 48 or not shadow_done.is_set():
        started = time.perf_counter()
        engine.active_step(seq, audio_frame(pcm, seq, engine.frame_size))
        concurrent.append((time.perf_counter() - started) * 1000)
        seq += 1
        deadline += 1 / FRAME_RATE
        time.sleep(max(0.0, deadline - time.perf_counter()))
    output.put(
        {
            "role": "active",
            "load_s": load_s,
            "warmup_s": warmup_s,
            "assigned_cpus": assigned_cpus,
            "baseline": latency_summary(baseline),
            "during_shadow_replay": latency_summary(concurrent),
        }
    )


def shadow_worker(
    model: dict[str, str],
    voice: str,
    input_audio: str,
    frames: int,
    model_ready: mp.synchronize.Event,
    all_models_ready: mp.synchronize.Event,
    baseline_done: mp.synchronize.Event,
    trace_queue: mp.Queue,
    trace_ready: mp.synchronize.Event,
    concurrent_start: mp.synchronize.Event,
    shadow_done: mp.synchronize.Event,
    output: mp.Queue,
) -> None:
    assigned_cpus = pin_worker(1)
    del input_audio
    load_started = time.perf_counter()
    engine = make_engine("cuda:1", "shadow-process", model)
    load_s = time.perf_counter() - load_started
    warm_started = time.perf_counter()
    engine.warmup()
    warmup_s = time.perf_counter() - warm_started
    model_ready.set()
    all_models_ready.wait()
    baseline_done.wait()

    role = "You are a helpful service agent."
    trace = trace_queue.get(timeout=60)
    engine.start_session(
        role + "\n\n<conversation_memory>Customer prefers Aurora.</conversation_memory>",
        voice,
    )
    trace_ready.set()
    concurrent_start.wait()
    replay_started = time.perf_counter()
    replay = engine.replay(trace)
    replay_wall_s = time.perf_counter() - replay_started
    shadow_done.set()
    output.put(
        {
            "role": "shadow",
            "load_s": load_s,
            "warmup_s": warmup_s,
            "assigned_cpus": assigned_cpus,
            "replay": {
                "frames": frames,
                "audio_s": frames / FRAME_RATE,
                "wall_s": replay_wall_s,
                "rtf": replay_wall_s / (frames / FRAME_RATE),
                "frames_per_s": frames / replay_wall_s,
                "user_code_mismatches": replay.user_code_mismatches,
                "user_code_values_differ": replay.user_code_values_differ,
                "user_codebook_mismatches": replay.user_codebook_mismatches,
                "target_mismatches": replay.target_mismatches,
                "decoder_mismatches": replay.decoder_mismatches,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--voice", default="NATF2.pt")
    parser.add_argument("--input-audio", default="assets/test/input_service.wav")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two visible GPUs are required")
    archive = Path(hf_hub_download(args.hf_repo, "voices.tgz"))
    model = {
        "mimi": hf_hub_download(args.hf_repo, loaders.MIMI_NAME),
        "moshi": hf_hub_download(args.hf_repo, loaders.MOSHI_NAME),
        "tokenizer": hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME),
    }
    voice = str(archive.parent / "voices" / args.voice)

    ctx = mp.get_context("spawn")
    active_ready = ctx.Event()
    shadow_ready = ctx.Event()
    all_models_ready = ctx.Event()
    baseline_done = ctx.Event()
    trace_queue = ctx.Queue()
    trace_ready = ctx.Event()
    concurrent_start = ctx.Event()
    shadow_done = ctx.Event()
    output = ctx.Queue()
    common = (
        model,
        voice,
        args.input_audio,
        args.frames,
    )
    active = ctx.Process(
        target=active_worker,
        args=(
            *common,
            active_ready,
            all_models_ready,
            baseline_done,
            trace_queue,
            trace_ready,
            concurrent_start,
            shadow_done,
            output,
        ),
    )
    shadow = ctx.Process(
        target=shadow_worker,
        args=(
            *common,
            shadow_ready,
            all_models_ready,
            baseline_done,
            trace_queue,
            trace_ready,
            concurrent_start,
            shadow_done,
            output,
        ),
    )
    active.start()
    shadow.start()
    active_ready.wait()
    shadow_ready.wait()
    all_models_ready.set()
    trace_ready.wait()
    concurrent_start.set()
    rows = [output.get(timeout=180), output.get(timeout=180)]
    active.join(timeout=30)
    shadow.join(timeout=30)
    if active.exitcode != 0 or shadow.exitcode != 0:
        raise RuntimeError(
            f"worker failure: active={active.exitcode}, shadow={shadow.exitcode}"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "one OS process per resident PersonaPlex GPU",
        "devices": [torch.cuda.get_device_name(index) for index in range(2)],
        "rows": {row["role"]: row for row in rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
