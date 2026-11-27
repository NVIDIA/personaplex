#!/usr/bin/env python3
"""Standalone PersonaPlex concurrency sweep. Edit the sweep globals, then run:

    python personaplex_benchmark.py --audio input.flac --output-jsonl results.jsonl

Requires Python 3.11+, torch/torchaudio, numpy, and websockets, and audio supported by torchaudio.load.
Mono/stereo input is downmixed and resampled to 24 kHz. Start the server
separately with admission capacity >= max(CONCURRENCY_LEVELS). No project imports.
Warmup runs one request per worker at the largest concurrency; measured levels
use rolling workers with the same workload. Progress and percentiles go to stderr;
JSONL is written directly to the selected file (replaced on each successful warmup).

Completion means input upload AND output-duration coverage, not verified source
completion: PersonaPlex may generate silence continuation. Input is paced in real
time. RTF is elapsed time divided by received audio duration: lower is better,
and 1 means real time. stream_rtf measures from the first output audio delta,
includes packet overshoot, and excludes cleanup. Inter-delta timings start
at the second nonempty audio delta and exclude cleanup output.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torchaudio
import websockets

# Configuration: keep these identical between the two server variants.
URL = "ws://127.0.0.1:8002/v1/realtime?duplex=1"
MODEL = "nvidia/personaplex-7b-v1"  # Must match the model name served by the endpoint.
AUDIO_FILE = Path("input.wav")
OUTPUT_JSONL = Path("/tmp/personaplex-results.jsonl")
CONCURRENCY_LEVELS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96)
REQUESTS_PER_WORKER = 2
START_JITTER_S = 5.0
TAIL_FRAMES = 13
REQUEST_TIMEOUT_S = 240.0
CLOSE_TIMEOUT_S = 10.0
VOICE = "NATF2.pt"
PERSONA = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920  # 80 ms per input append; float32 in, PCM16 out.
QUANTILES = (0.0, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.99, 1.0)


def parse_concurrency_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers, e.g. 1,2,4,8") from exc
    if any(level <= 0 for level in levels):
        raise argparse.ArgumentTypeError("Concurrency levels must be positive integers")
    return levels


def parse_args(argv=None):
    """Configure the model, input audio, concurrency sweep, and JSONL destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="Model name/path served by the endpoint")
    parser.add_argument("--audio", type=Path, default=AUDIO_FILE, help="Input audio, e.g. WAV or FLAC")
    parser.add_argument("--output-jsonl", type=Path, default=OUTPUT_JSONL, help="Output JSONL file (overwritten)")
    parser.add_argument(
        "--concurrency-levels",
        type=parse_concurrency_levels,
        default=CONCURRENCY_LEVELS,
        metavar="N,N,...",
        help="Comma-separated positive concurrency levels (default: %(default)s)",
    )
    return parser.parse_args(argv)


def load_audio() -> bytes:
    """Use the original driver's mono float32 conversion and resampling."""
    waveform, rate = torchaudio.load(str(AUDIO_FILE))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)
    pcm = waveform.reshape(-1).cpu().float().numpy().astype("<f4", copy=False).tobytes()
    if not pcm:
        raise ValueError("Input audio is empty")
    return pcm


async def measure(pcm: bytes) -> dict:
    """Measure one fresh session, retaining failures and completed timing on close errors."""
    samples = len(pcm) // 4
    target = (math.ceil(samples / FRAME_SAMPLES) + TAIL_FRAMES) * FRAME_SAMPLES
    # A successor frame flushes the connector; it isn't part of the coverage target.
    segments = (pcm, bytes((target - samples + FRAME_SAMPLES) * 4))
    started = time.perf_counter()
    stream_started = coverage_at = first_audio = None
    last_send = started
    output_bytes = audio_chunks = sent_samples = 0
    previous_audio = None
    audio_inter_delta_ms = []
    reached = asyncio.Event()
    closing = False
    phase = "connect"
    result = {}
    try:
        async with (
            asyncio.timeout(REQUEST_TIMEOUT_S),
            websockets.connect(
                URL,
                compression=None,
                ping_interval=30,
                ping_timeout=300,
                close_timeout=CLOSE_TIMEOUT_S,
                max_size=64 * 1024 * 1024,
            ) as ws,
        ):
            phase = "admission"
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "model": MODEL,
                            "modalities": ["audio", "text"],  # Original session shape; text events are ignored.
                            "input_audio_format": "pcm_f32le",
                            "sample_rate_hz": SAMPLE_RATE,
                            "output_audio_format": "pcm16",
                            "turn_detection": None,
                            "voice": VOICE,
                            "instructions": PERSONA,
                            "extra_body": {"auto_response": True},
                        },
                    }
                )
            )
            created = json.loads(await ws.recv())
            if created.get("type") != "session.created":
                raise RuntimeError(f"Expected session.created: {created}")

            async def send_audio():
                nonlocal stream_started, last_send, sent_samples, phase
                phase = "upload"
                for segment in segments:
                    for offset in range(0, len(segment), FRAME_SAMPLES * 4):
                        chunk = segment[offset : offset + FRAME_SAMPLES * 4]
                        if stream_started is None:
                            stream_started = time.perf_counter()
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "format": "pcm_f32le",
                                    "sample_rate_hz": SAMPLE_RATE,
                                    "audio": base64.b64encode(chunk).decode("ascii"),
                                }
                            )
                        )
                        last_send = time.perf_counter()
                        sent_samples += len(chunk) // 4
                        # Account for send work and sleep overshoot across the whole stream.
                        deadline = stream_started + sent_samples / SAMPLE_RATE
                        delay_s = max(0.0, deadline - time.perf_counter())
                        if delay_s > 0:
                            await asyncio.sleep(delay_s)
                phase = "coverage"

            async def receive_audio():
                nonlocal output_bytes, audio_chunks, first_audio, coverage_at, previous_audio
                while True:
                    event = json.loads(await ws.recv())
                    kind = event.get("type")
                    if kind == "error":
                        raise RuntimeError(f"Server rejected request: {event}")
                    if kind == "session.closed":
                        if not closing:
                            raise RuntimeError("Session closed before upload/coverage completed")
                        return
                    if kind not in {"response.output_audio.delta", "response.audio.delta"}:
                        continue
                    if (
                        event.get("format", "pcm16") not in {"pcm16", "pcm", "pcm_s16le", "s16le"}
                        or event.get("sample_rate_hz", SAMPLE_RATE) != SAMPLE_RATE
                    ):
                        raise ValueError("Expected 24 kHz PCM16 output")
                    chunk = base64.b64decode(event.get("delta", ""), validate=True)
                    if len(chunk) % 2:
                        raise ValueError("Incomplete PCM16 output sample")
                    if chunk:
                        now = time.perf_counter()
                        output_bytes += len(chunk)
                        audio_chunks += 1
                        if first_audio is None:
                            first_audio = now
                        if previous_audio is not None:
                            audio_inter_delta_ms.append((now - previous_audio) * 1000)
                        previous_audio = now
                        if output_bytes // 2 >= target and coverage_at is None:
                            coverage_at = now
                            reached.set()

            async def complete():
                await sender
                await reached.wait()

            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(receive_audio())
            completion = asyncio.create_task(complete())
            try:
                done, _ = await asyncio.wait((completion, receiver), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if not completion.done():
                    raise RuntimeError("Stream ended before upload/coverage completed")
                # Include every counted audio delta, including those after coverage.
                end = max(last_send, previous_audio)
                duration_s = output_bytes / (SAMPLE_RATE * 2)
                result = {
                    "elapsed_ms": (end - started) * 1000,
                    "total_processing_ms": (end - stream_started) * 1000,
                    "first_audio_delta_ms": (first_audio - started) * 1000,
                    "final_output_lag_ms": max(0, coverage_at - last_send) * 1000,
                    "output_audio_s": duration_s,
                    "audio_chunks": audio_chunks,
                    "audio_inter_delta_ms": list(audio_inter_delta_ms),
                    "stream_rtf": (end - first_audio) / duration_s,
                }
                # Freeze metrics before close can flush additional output.
                closing, phase = True, "close"
                await ws.send(json.dumps({"type": "session.close"}))
                await asyncio.wait_for(receiver, CLOSE_TIMEOUT_S)
            finally:
                for task in (sender, receiver, completion):
                    task.cancel()
                await asyncio.gather(sender, receiver, completion, return_exceptions=True)
        return {"ok": True, **result}
    except Exception as exc:
        return {
            "ok": False,
            **result,
            "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "sent_samples": sent_samples,
            "received_samples": output_bytes // 2,
            "expected_samples": target,
            "coverage_reached": reached.is_set(),
            "audio_inter_delta_ms": result.get("audio_inter_delta_ms", list(audio_inter_delta_ms)),
        }


async def run_level(pcm: bytes, concurrency: int, repeats: int, *, jitter: bool = True) -> list[dict]:
    """Keep a fixed number of rolling workers; count every attempted request."""

    async def worker():
        rows = []
        for _ in range(repeats):
            if jitter and START_JITTER_S:
                await asyncio.sleep(random.uniform(0, START_JITTER_S))
            rows.append(await measure(pcm))
        return rows

    return [row for rows in await asyncio.gather(*(worker() for _ in range(concurrency))) for row in rows]


def summarize(values: list[float]) -> dict:
    """Return empty statistics when a level has no successful requests."""
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "max": max(values),
        "percentiles": {
            f"p{q * 100:g}": float(value) for q, value in zip(QUANTILES, np.quantile(values, QUANTILES), strict=True)
        },
    }


def print_latency(label: str, stats: dict) -> None:
    """Print the original driver's latency quantiles, with linear interpolation."""
    if not stats:
        print(f"{label}: no successful requests", file=sys.stderr)
        return
    percentiles = stats["percentiles"]
    span = percentiles["p84"] - percentiles["p16"]
    print(
        f"{label}: avg={round(stats['mean']):,} ms std={round(stats['std']):,} ms span p16..p84={round(span):,} ms",
        file=sys.stderr,
    )
    for q in QUANTILES:
        print(f"  q={q:>4.2f}: {round(percentiles[f'p{q * 100:g}']):,} ms", file=sys.stderr)


async def main():
    if (
        not CONCURRENCY_LEVELS
        or any(n <= 0 for n in CONCURRENCY_LEVELS)
        or REQUESTS_PER_WORKER <= 0
        or TAIL_FRAMES < 0
        or START_JITTER_S < 0
        or REQUEST_TIMEOUT_S <= 0
        or CLOSE_TIMEOUT_S <= 0
    ):
        raise ValueError("Invalid benchmark configuration")
    if OUTPUT_JSONL.resolve() == AUDIO_FILE.resolve():
        raise ValueError("Output JSONL must not overwrite the input audio")
    pcm = load_audio()
    config = {
        "url": URL,
        "model": MODEL,
        "voice": VOICE,
        "persona": PERSONA,
        "input_sha256": hashlib.sha256(AUDIO_FILE.read_bytes()).hexdigest(),
        "input_audio_s": len(pcm) / (SAMPLE_RATE * 4),
        "tail_frames": TAIL_FRAMES,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "requests_per_worker": REQUESTS_PER_WORKER,
        "start_jitter_s": START_JITTER_S,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "close_timeout_s": CLOSE_TIMEOUT_S,
        "compression": None,
        "completion_mode": "audio_coverage",
        "stream_rtf_definition": "seconds from first output audio delta to completion / received audio seconds",
        "input_completion_verified": False,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "torchaudio", "websockets")},
    }
    print(f"Warmup: {max(CONCURRENCY_LEVELS)} sessions", file=sys.stderr)
    warmup = await run_level(pcm, max(CONCURRENCY_LEVELS), 1, jitter=False)
    if any(not row["ok"] for row in warmup):
        raise RuntimeError(f"Warmup failed: {[row for row in warmup if not row['ok']]}")
    failed = False
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as output:
        for concurrency in CONCURRENCY_LEVELS:
            print(f"N={concurrency}: {concurrency * REQUESTS_PER_WORKER} requests", file=sys.stderr)
            rows = await run_level(pcm, concurrency, REQUESTS_PER_WORKER)
            good = [row for row in rows if row["ok"]]
            failed |= len(good) != len(rows)
            summary = {
                key: summarize([row[key] for row in good])
                for key in (
                    "stream_rtf",
                    "total_processing_ms",
                    "elapsed_ms",
                    "first_audio_delta_ms",
                    "final_output_lag_ms",
                    "output_audio_s",
                )
            }
            summary["audio_inter_delta_ms"] = summarize(
                [interval for row in good for interval in row["audio_inter_delta_ms"]]
            )
            print(
                json.dumps(
                    {
                        "config": config,
                        "concurrency": concurrency,
                        "successes": len(good),
                        "failures": len(rows) - len(good),
                        "summary": summary,
                        "requests": rows,
                    },
                    allow_nan=False,
                ),
                file=output,
                flush=True,
            )
            print_latency("First input to coverage", summary["total_processing_ms"])
            print_latency("First audio delta latency", summary["first_audio_delta_ms"])
            print_latency("Inter-delta audio latency", summary["audio_inter_delta_ms"])
            print(
                f"N={concurrency}: {len(good)}/{len(rows)} succeeded; stream RTF {summary['stream_rtf'].get('median')}",
                file=sys.stderr,
            )
    if failed:
        raise SystemExit("One or more requests failed; inspect the JSONL failure rows")


if __name__ == "__main__":
    args = parse_args()
    MODEL, AUDIO_FILE, OUTPUT_JSONL = args.model, args.audio, args.output_jsonl
    CONCURRENCY_LEVELS = args.concurrency_levels
    asyncio.run(main())
