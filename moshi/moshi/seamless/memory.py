"""GPU-1 memory-worker contract used by the seamless gateway."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

import aiohttp
import numpy as np

from .types import CompactRequest, CompactResult


class MemoryWorker(Protocol):
    async def start_session(self, session_id: str, role_prompt: str) -> None: ...

    async def push_audio(self, session_id: str, seq: int, pcm: np.ndarray) -> None: ...

    async def push_assistant_text(
        self, session_id: str, seq: int, text_piece: str
    ) -> None: ...

    async def compact(self, request: CompactRequest) -> CompactResult: ...

    async def close_session(self, session_id: str) -> None: ...


class NullMemoryWorker:
    """Manual-install mode: audio is ignored and auto compaction is disabled."""

    auto_compaction = False

    async def start_session(self, session_id: str, role_prompt: str) -> None:
        del session_id, role_prompt

    async def push_audio(self, session_id: str, seq: int, pcm: np.ndarray) -> None:
        del session_id, seq, pcm

    async def push_assistant_text(
        self, session_id: str, seq: int, text_piece: str
    ) -> None:
        del session_id, seq, text_piece

    async def compact(self, request: CompactRequest) -> CompactResult:
        raise RuntimeError("automatic compaction requires --memory-worker-url")

    async def close_session(self, session_id: str) -> None:
        del session_id


class HttpMemoryWorker:
    """Non-blocking client for a streaming-ASR + compactor service.

    The service contract is documented in ``docs/seamless_install.md``. Audio
    uploads are serialized through a bounded queue so ASR latency cannot stall
    the PersonaPlex audio loop. Queue overflow is fatal: silently dropping ASR
    evidence would make a later compact cutoff unauditable.
    """

    auto_compaction = True

    def __init__(
        self,
        base_url: str,
        *,
        sample_rate: int = 24_000,
        queue_frames: int = 256,
        request_timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sample_rate = sample_rate
        self._queue_frames = queue_frames
        self._queue: asyncio.Queue[tuple[str, str, int, bytes | str] | None] = (
            asyncio.Queue(maxsize=queue_frames)
        )
        self._session: aiohttp.ClientSession | None = None
        self._sender: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s)

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def start_session(self, session_id: str, role_prompt: str) -> None:
        if self._sender is not None and not self._sender.done():
            raise RuntimeError("memory worker already has an active session")
        if self._failure is not None or self._queue.qsize():
            self._queue = asyncio.Queue(maxsize=self._queue_frames)
        client = await self._client()
        async with client.post(
            f"{self.base_url}/v1/session/start",
            json={"session_id": session_id, "role_prompt": role_prompt},
        ) as response:
            await self._require_success(response)
        self._failure = None
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(
                self._audio_sender(), name="seamless-memory-audio-sender"
            )

    async def push_audio(self, session_id: str, seq: int, pcm: np.ndarray) -> None:
        if self._failure is not None:
            raise RuntimeError("memory audio sender failed") from self._failure
        payload = np.asarray(pcm, dtype="<f4").reshape(-1).tobytes()
        try:
            self._queue.put_nowait(("user_audio", session_id, seq, payload))
        except asyncio.QueueFull as exc:
            raise RuntimeError(
                "memory worker audio queue overflow; refusing evidence loss"
            ) from exc

    async def push_assistant_text(
        self, session_id: str, seq: int, text_piece: str
    ) -> None:
        if not text_piece:
            return
        if self._failure is not None:
            raise RuntimeError("memory event sender failed") from self._failure
        try:
            self._queue.put_nowait(("assistant_text", session_id, seq, text_piece))
        except asyncio.QueueFull as exc:
            raise RuntimeError(
                "memory worker event queue overflow; refusing evidence loss"
            ) from exc

    async def compact(self, request: CompactRequest) -> CompactResult:
        if self._failure is not None:
            raise RuntimeError("memory audio sender failed") from self._failure
        join_task = asyncio.create_task(self._queue.join())
        sender = self._sender
        if sender is not None:
            done, _ = await asyncio.wait(
                (join_task, sender), return_when=asyncio.FIRST_COMPLETED
            )
            if sender in done and not join_task.done():
                join_task.cancel()
                await asyncio.gather(join_task, return_exceptions=True)
                raise RuntimeError("memory audio sender failed") from self._failure
        await join_task
        if self._failure is not None:
            raise RuntimeError("memory audio sender failed") from self._failure
        client = await self._client()
        body = {
            "session_id": request.session_id,
            "request_id": request.request_id,
            "parent_version": request.parent_version,
            "cutoff_hint_seq": request.cutoff_hint_seq,
            "previous_memory": request.previous_memory,
            "role_prompt": request.role_prompt,
            "ledger_digest": request.ledger_digest,
        }
        async with client.post(
            f"{self.base_url}/v1/session/compact", json=body
        ) as response:
            payload = await self._require_success(response)
        return CompactResult(
            request_id=str(payload["request_id"]),
            parent_version=int(payload["parent_version"]),
            cutoff_seq=int(payload["cutoff_seq"]),
            memory=str(payload["memory"]),
            metadata=dict(payload.get("metadata", {})),
        )

    async def close_session(self, session_id: str) -> None:
        if self._sender is not None:
            if not self._sender.done():
                await self._queue.put(None)
            try:
                await self._sender
            except Exception:
                # The operational failure was already surfaced by push/compact.
                pass
            finally:
                self._sender = None
        if self._session is not None and not self._session.closed:
            try:
                async with self._session.post(
                    f"{self.base_url}/v1/session/close",
                    json={"session_id": session_id},
                ) as response:
                    await self._require_success(response)
            finally:
                await self._session.close()
                self._session = None

    async def _audio_sender(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                try:
                    if item is None:
                        return
                    kind, session_id, seq, payload = item
                    client = await self._client()
                    if kind == "user_audio":
                        headers = {
                            "X-Session-Id": session_id,
                            "X-Frame-Seq": str(seq),
                            "X-Sample-Rate": str(self.sample_rate),
                            "Content-Type": "audio/f32le",
                        }
                        async with client.post(
                            f"{self.base_url}/v1/session/audio",
                            data=payload,
                            headers=headers,
                        ) as response:
                            await self._require_success(response)
                    elif kind == "assistant_text":
                        async with client.post(
                            f"{self.base_url}/v1/session/assistant",
                            json={
                                "session_id": session_id,
                                "frame_seq": seq,
                                "text_piece": str(payload),
                            },
                        ) as response:
                            await self._require_success(response)
                    else:
                        raise RuntimeError(f"unknown memory event kind: {kind}")
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._failure = exc
            raise

    @staticmethod
    async def _require_success(response: aiohttp.ClientResponse) -> dict:
        text = await response.text()
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(
                f"memory worker returned HTTP {response.status}: {text[:500]}"
            )
        if not text.strip():
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("memory worker returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("memory worker response must be a JSON object")
        return value
