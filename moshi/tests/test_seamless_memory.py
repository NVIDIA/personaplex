from __future__ import annotations

import unittest

from aiohttp import web
import numpy as np

from moshi.seamless.memory import HttpMemoryWorker
from moshi.seamless.types import CompactRequest


class HttpMemoryWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.events: list[tuple[str, object]] = []
        app = web.Application()

        async def start(request: web.Request) -> web.Response:
            self.events.append(("start", await request.json()))
            return web.json_response({})

        async def audio(request: web.Request) -> web.Response:
            self.events.append(
                (
                    "audio",
                    {
                        "seq": request.headers["X-Frame-Seq"],
                        "body": await request.read(),
                    },
                )
            )
            return web.json_response({})

        async def assistant(request: web.Request) -> web.Response:
            self.events.append(("assistant", await request.json()))
            return web.json_response({})

        async def compact(request: web.Request) -> web.Response:
            payload = await request.json()
            self.events.append(("compact", payload))
            return web.json_response(
                {
                    "request_id": payload["request_id"],
                    "parent_version": payload["parent_version"],
                    "cutoff_seq": payload["cutoff_hint_seq"],
                    "memory": "User | project = Aurora",
                }
            )

        async def close(request: web.Request) -> web.Response:
            self.events.append(("close", await request.json()))
            return web.json_response({})

        app.router.add_post("/v1/session/start", start)
        app.router.add_post("/v1/session/audio", audio)
        app.router.add_post("/v1/session/assistant", assistant)
        app.router.add_post("/v1/session/compact", compact)
        app.router.add_post("/v1/session/close", close)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        socket = self.site._server.sockets[0]
        self.base_url = f"http://127.0.0.1:{socket.getsockname()[1]}"

    async def asyncTearDown(self) -> None:
        await self.runner.cleanup()

    async def test_compact_waits_for_ordered_audio_and_assistant_events(self) -> None:
        worker = HttpMemoryWorker(self.base_url, queue_frames=8)
        await worker.start_session("s1", "Role")
        pcm = np.arange(4, dtype=np.float32)
        await worker.push_audio("s1", 7, pcm)
        await worker.push_assistant_text("s1", 3, " Aurora")
        result = await worker.compact(
            CompactRequest(
                session_id="s1",
                request_id="r1",
                parent_version=0,
                cutoff_hint_seq=7,
                previous_memory="",
                role_prompt="Role",
                ledger_digest="digest",
            )
        )
        await worker.close_session("s1")

        self.assertEqual(result.cutoff_seq, 7)
        self.assertEqual(
            [event for event, _ in self.events],
            ["start", "audio", "assistant", "compact", "close"],
        )
        self.assertEqual(self.events[1][1]["body"], pcm.astype("<f4").tobytes())
        self.assertEqual(self.events[2][1]["frame_seq"], 3)


if __name__ == "__main__":
    unittest.main()
