"""Local-loopback transport benchmark for HttpMemoryWorker."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

from aiohttp import web
import numpy as np

from moshi.seamless.memory import HttpMemoryWorker
from moshi.seamless.types import CompactRequest


def summary_ms(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


async def benchmark(output: Path, batches: list[int]) -> None:
    received = 0
    app = web.Application()

    async def ok(request: web.Request) -> web.Response:
        nonlocal received
        if request.path.endswith("/audio"):
            await request.read()
            received += 1
        elif request.path.endswith("/assistant"):
            await request.json()
            received += 1
        else:
            await request.json()
        return web.json_response({})

    async def compact(request: web.Request) -> web.Response:
        payload = await request.json()
        return web.json_response(
            {
                "request_id": payload["request_id"],
                "parent_version": payload["parent_version"],
                "cutoff_seq": payload["cutoff_hint_seq"],
                "memory": "memory",
            }
        )

    app.router.add_post("/v1/session/start", ok)
    app.router.add_post("/v1/session/audio", ok)
    app.router.add_post("/v1/session/assistant", ok)
    app.router.add_post("/v1/session/compact", compact)
    app.router.add_post("/v1/session/close", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    worker = HttpMemoryWorker(f"http://127.0.0.1:{port}", queue_frames=4096)
    pcm = np.zeros(1920, dtype=np.float32)
    rows = []
    seq = 0
    try:
        started = time.perf_counter()
        await worker.start_session("bench", "Role")
        start_ms = (time.perf_counter() - started) * 1000
        for batch in batches:
            enqueue_ms = []
            received_before = received
            batch_started = time.perf_counter()
            for _ in range(batch):
                begin = time.perf_counter()
                await worker.push_audio("bench", seq, pcm)
                enqueue_ms.append((time.perf_counter() - begin) * 1000)
                if seq % 5 == 0:
                    begin = time.perf_counter()
                    await worker.push_assistant_text("bench", seq, " token")
                    enqueue_ms.append((time.perf_counter() - begin) * 1000)
                seq += 1
            enqueue_wall_s = time.perf_counter() - batch_started
            drain_started = time.perf_counter()
            await worker.compact(
                CompactRequest(
                    session_id="bench",
                    request_id=f"r-{seq}",
                    parent_version=0,
                    cutoff_hint_seq=seq - 1,
                    previous_memory="",
                    role_prompt="Role",
                    ledger_digest="digest",
                )
            )
            drain_wall_s = time.perf_counter() - drain_started
            event_count = received - received_before
            rows.append(
                {
                    "audio_frames": batch,
                    "events": event_count,
                    "enqueue_call_ms": summary_ms(enqueue_ms),
                    "enqueue_wall_s": enqueue_wall_s,
                    "drain_and_compact_wall_s": drain_wall_s,
                    "transport_events_per_s": event_count / drain_wall_s,
                    "realtime_required_audio_frames_per_s": 12.5,
                }
            )
        await worker.close_session("bench")
        payload = {
            "schema_version": 1,
            "method": "aiohttp loopback memory-worker transport burst",
            "session_start_ms": start_ms,
            "rows": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    finally:
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128, 512])
    args = parser.parse_args()
    asyncio.run(benchmark(args.output, args.batches))


if __name__ == "__main__":
    main()
