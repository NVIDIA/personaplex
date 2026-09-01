"""PersonaPlex WebDemo gateway with shadow or inplace memory installation."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import logging
import os
from pathlib import Path
import random
import tarfile
import time
from typing import Any
import uuid

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sphn
import torch

from .models import loaders
from .seamless.controller import SeamlessInstallController
from .seamless.memory import HttpMemoryWorker, NullMemoryWorker
from .seamless.model_slot import (
    AsyncPersonaPlexSlot,
    PersonaPlexSlotEngine,
    ProcessPersonaPlexSlot,
)
from .seamless.types import ControllerConfig, InstallBackend
from .utils.connection import create_ssl_context, get_lan_ip


logger = logging.getLogger(__name__)


def seed_all(seed: int, *, include_cuda: bool = True) -> None:
    # ``torch.manual_seed`` also touches CUDA generators. Keep the gateway
    # process CUDA-free when models are isolated in children.
    torch.random.default_generator.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _get_voice_prompt_dir(voice_prompt_dir: str | None, hf_repo: str) -> str:
    if voice_prompt_dir is not None:
        directory = Path(voice_prompt_dir)
    else:
        archive = Path(hf_hub_download(hf_repo, "voices.tgz"))
        directory = archive.parent / "voices"
        if not directory.exists():
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=archive.parent)
    if not directory.is_dir():
        raise FileNotFoundError(f"voice prompt directory not found: {directory}")
    return str(directory)


def _get_static_path(static: str | None) -> str | None:
    if static == "none":
        return None
    if static is not None:
        path = Path(static)
    else:
        archive = Path(hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz"))
        path = archive.parent / "dist"
        if not path.exists():
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=archive.parent)
    if not path.is_dir():
        raise FileNotFoundError(f"static WebDemo directory not found: {path}")
    return str(path)


class SeamlessWebState:
    """Own one browser socket across either supported install backend."""

    def __init__(
        self,
        *,
        slots: tuple[Any, ...],
        controller: SeamlessInstallController,
        voice_prompt_dir: str,
        sample_rate: int,
        frame_size: int,
        control_token: str | None,
    ) -> None:
        self.slots = slots
        self.controller = controller
        self.voice_prompt_dir = Path(voice_prompt_dir).resolve()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.control_token = control_token
        self.connection_lock = asyncio.Lock()
        self.frame_latencies_ms: deque[float] = deque(maxlen=20_000)
        self.session_started_at: float | None = None

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if self.connection_lock.locked():
            await ws.close(code=1013, message=b"PersonaPlex session is busy")
            return ws

        role_prompt = request.query.get("text_prompt", "").strip()
        voice_name = request.query.get("voice_prompt", "")
        try:
            voice_prompt = self._resolve_voice_prompt(voice_name)
        except (ValueError, FileNotFoundError) as exc:
            await ws.close(code=1008, message=str(exc).encode("utf-8")[:120])
            return ws
        seed = int(request.query.get("seed", "-1"))
        session_id = uuid.uuid4().hex
        peer = request.remote or "unknown"
        logger.info("incoming seamless session %s from %s", session_id, peer)

        async with self.connection_lock:
            if seed >= 0:
                seed_all(seed, include_cuda=False)
                await asyncio.gather(*(slot.set_seed(seed) for slot in self.slots))
            close = False
            opus_writer = sphn.OpusStreamWriter(self.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.sample_rate)
            try:
                self.frame_latencies_ms.clear()
                self.session_started_at = time.time()
                await self.controller.start_session(
                    session_id=session_id,
                    role_prompt=role_prompt,
                    voice_prompt=voice_prompt,
                )
                await ws.send_bytes(b"\x00")
                logger.info(
                    "session %s ready on %s",
                    session_id,
                    self.controller.active_slot.name,
                )

                async def recv_loop() -> None:
                    nonlocal close
                    try:
                        async for message in ws:
                            if message.type in (
                                aiohttp.WSMsgType.ERROR,
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.CLOSE,
                            ):
                                return
                            if message.type != aiohttp.WSMsgType.BINARY:
                                logger.warning(
                                    "unexpected websocket message %s", message.type
                                )
                                continue
                            payload = message.data
                            if not isinstance(payload, bytes) or not payload:
                                continue
                            if payload[0] == 1:
                                opus_reader.append_bytes(payload[1:])
                            else:
                                logger.warning(
                                    "unknown client message kind %d", payload[0]
                                )
                    finally:
                        close = True

                async def inference_loop() -> None:
                    buffered = np.empty(0, dtype=np.float32)
                    while not close:
                        await asyncio.sleep(0.001)
                        pcm = np.asarray(
                            opus_reader.read_pcm(), dtype=np.float32
                        ).reshape(-1)
                        if pcm.size:
                            buffered = np.concatenate((buffered, pcm))
                        while buffered.size >= self.frame_size:
                            chunk = buffered[: self.frame_size]
                            buffered = buffered[self.frame_size :]
                            started = time.monotonic()
                            result = await self.controller.process_pcm(chunk)
                            if result.output_pcm is not None:
                                opus_writer.append_pcm(result.output_pcm)
                            if result.output_text:
                                await ws.send_bytes(
                                    b"\x02" + result.output_text.encode("utf-8")
                                )
                            elapsed = time.monotonic() - started
                            self.frame_latencies_ms.append(elapsed * 1000)
                            frame_seconds = self.frame_size / self.sample_rate
                            if elapsed > frame_seconds:
                                logger.warning(
                                    "active frame %d missed realtime budget: %.1f ms",
                                    result.frame.seq,
                                    elapsed * 1000,
                                )

                async def send_loop() -> None:
                    while not close:
                        await asyncio.sleep(0.001)
                        payload = opus_writer.read_bytes()
                        if payload:
                            await ws.send_bytes(b"\x01" + payload)
                        for event in self.controller.drain_events():
                            logger.info(
                                "seamless event %s", json.dumps(event, sort_keys=True)
                            )

                tasks = [
                    asyncio.create_task(recv_loop(), name="seamless-ws-recv"),
                    asyncio.create_task(inference_loop(), name="seamless-inference"),
                    asyncio.create_task(send_loop(), name="seamless-ws-send"),
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    error = task.exception()
                    if error is not None:
                        logger.exception(
                            "session loop failed",
                            exc_info=(type(error), error, error.__traceback__),
                        )
            except Exception:
                logger.exception("seamless session %s failed", session_id)
            finally:
                close = True
                try:
                    await self.controller.close_session()
                except Exception:
                    logger.exception("failed to close seamless session %s", session_id)
                await ws.close()
                logger.info("seamless session %s closed", session_id)
        return ws

    async def handle_status(self, request: web.Request) -> web.Response:
        del request
        status = self.controller.status()
        values = np.asarray(self.frame_latencies_ms, dtype=np.float64)
        frame_budget_ms = self.frame_size / self.sample_rate * 1000
        status["active_latency"] = {
            "frames": int(values.size),
            "frame_budget_ms": frame_budget_ms,
            "p50_ms": float(np.percentile(values, 50)) if values.size else None,
            "p95_ms": float(np.percentile(values, 95)) if values.size else None,
            "p99_ms": float(np.percentile(values, 99)) if values.size else None,
            "max_ms": float(values.max()) if values.size else None,
            "deadline_miss_rate": (
                float(np.mean(values > frame_budget_ms)) if values.size else None
            ),
            "session_wall_s": (
                time.time() - self.session_started_at
                if self.session_started_at is not None
                else None
            ),
        }
        return web.json_response(status)

    async def handle_install(self, request: web.Request) -> web.Response:
        self._authorize_control(request)
        if self.controller.session_id is None:
            raise web.HTTPConflict(text="no active session")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, TypeError) as exc:
            raise web.HTTPBadRequest(text="request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="request body must be a JSON object")
        memory = payload.get("memory")
        if memory is not None and (not isinstance(memory, str) or not memory.strip()):
            raise web.HTTPBadRequest(
                text="memory must be a non-empty string when provided"
            )
        cutoff = payload.get("cutoff_seq")
        if cutoff is not None and not isinstance(cutoff, int):
            raise web.HTTPBadRequest(text="cutoff_seq must be an integer")
        try:
            accepted = await self.controller.request_install(
                memory=memory,
                cutoff_seq=cutoff,
                reason="http_memory" if memory is not None else "http_trigger",
            )
        except (RuntimeError, ValueError) as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        return web.json_response(
            {"accepted": accepted, "status": self.controller.status()},
            status=202 if accepted else 409,
        )

    async def cleanup(self, app: web.Application) -> None:
        del app
        try:
            await self.controller.close_session()
        finally:
            for slot in self.slots:
                slot.close()

    def _resolve_voice_prompt(self, filename: str) -> str:
        if not filename:
            raise ValueError("voice_prompt query parameter is required")
        candidate = (self.voice_prompt_dir / filename).resolve()
        if self.voice_prompt_dir not in candidate.parents:
            raise ValueError("voice_prompt must stay inside voice-prompt-dir")
        if not candidate.is_file():
            raise FileNotFoundError(f"voice prompt not found: {filename}")
        return str(candidate)

    def _authorize_control(self, request: web.Request) -> None:
        if self.control_token is None:
            return
        if request.headers.get("Authorization") != f"Bearer {self.control_token}":
            raise web.HTTPUnauthorized(text="invalid control token")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PersonaPlex gateway with shadow or inplace memory installation"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--static")
    parser.add_argument("--ssl")
    parser.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--tokenizer")
    parser.add_argument("--moshi-weight")
    parser.add_argument("--mimi-weight")
    parser.add_argument("--voice-prompt-dir")
    parser.add_argument(
        "--install-backend",
        choices=tuple(backend.value for backend in InstallBackend),
        default=InstallBackend.SHADOW.value,
        help=(
            "shadow: two PersonaPlex GPUs and atomic handoff; "
            "inplace: one PersonaPlex GPU and a blocking batched M+W prefill"
        ),
    )
    parser.add_argument("--active-device", default="cuda:1")
    parser.add_argument(
        "--shadow-device",
        default="cuda:2",
        help="second PersonaPlex device used only by --install-backend shadow",
    )
    parser.add_argument(
        "--slot-runtime",
        choices=("process", "thread"),
        default="process",
        help="isolate each CUDA context in a process (production default)",
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--memory-worker-url")
    parser.add_argument(
        "--control-token",
        help="optional Bearer token required by POST /api/seamless/install",
    )
    parser.add_argument("--memory-queue-frames", type=int, default=256)
    parser.add_argument("--context-capacity-frames", type=int, default=3000)
    parser.add_argument("--trigger-reserve-frames", type=int, default=1400)
    parser.add_argument(
        "--prefill-batch-frames",
        "--replay-batch-frames",
        dest="prefill_batch_frames",
        type=int,
        default=64,
        help=(
            "duplex positions per batched Transformer prefill; "
            "--replay-batch-frames is a deprecated alias"
        ),
    )
    parser.add_argument("--text-prefill-chunk-tokens", type=int, default=64)
    parser.add_argument(
        "--inplace-min-tail-frames",
        type=int,
        default=256,
        help=(
            "minimum raw W accumulated after the compact cutoff before a "
            "destructive inplace prefill"
        ),
    )
    parser.add_argument("--switch-max-lag-frames", type=int, default=2)
    parser.add_argument("--safe-silence-frames", type=int, default=3)
    parser.add_argument("--silence-rms-threshold", type=float, default=0.004)
    parser.add_argument("--force-switch-after-s", type=float, default=8.0)
    parser.add_argument("--install-timeout-s", type=float, default=180.0)
    parser.add_argument("--codec-convergence-frames", type=int, default=16)
    parser.add_argument("--decoder-convergence-frames", type=int, default=16)
    parser.add_argument("--decoder-mse-threshold", type=float, default=1e-7)
    parser.add_argument(
        "--disable-auto-trigger",
        action="store_true",
        help="wait for an authenticated HTTP trigger instead of capacity policy",
    )
    parser.add_argument("--ledger-max-frames", type=int, default=20_000)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    if (
        args.install_backend == InstallBackend.SHADOW.value
        and args.active_device == args.shadow_device
    ):
        raise ValueError("active-device and shadow-device must be different")
    seed_all(42424242, include_cuda=args.slot_runtime == "thread")

    voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    static_path = _get_static_path(args.static)
    hf_hub_download(args.hf_repo, "config.json")
    mimi_weight = args.mimi_weight or hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    moshi_weight = args.moshi_weight or hf_hub_download(
        args.hf_repo, loaders.MOSHI_NAME
    )
    tokenizer = args.tokenizer or hf_hub_download(
        args.hf_repo, loaders.TEXT_TOKENIZER_NAME
    )

    async def make_app() -> web.Application:
        slot_specs = (("slot-a", args.active_device),)
        if args.install_backend == InstallBackend.SHADOW.value:
            slot_specs = (
                *slot_specs,
                ("slot-b", args.shadow_device),
            )
        if args.slot_runtime == "process":
            slots = tuple(
                ProcessPersonaPlexSlot(
                    name=name,
                    device=device,
                    mimi_weight=mimi_weight,
                    moshi_weight=moshi_weight,
                    tokenizer_path=tokenizer,
                    cpu_offload=args.cpu_offload,
                    decoder_mse_threshold=args.decoder_mse_threshold,
                    process_rank=process_rank,
                    process_world_size=len(slot_specs),
                )
                for process_rank, (name, device) in enumerate(slot_specs)
            )
            logger.info("loading and warming isolated resident processes")
            try:
                await asyncio.gather(*(slot.warmup() for slot in slots))
            except BaseException:
                for slot in slots:
                    slot.close()
                raise
            sample_rate = slots[0].sample_rate
            frame_size = slots[0].frame_size
        else:
            engines = []
            for name, device in slot_specs:
                logger.info("loading threaded resident %s on %s", name, device)
                engines.append(
                    PersonaPlexSlotEngine(
                        name=name,
                        device=device,
                        mimi_weight=mimi_weight,
                        moshi_weight=moshi_weight,
                        tokenizer_path=tokenizer,
                        cpu_offload=args.cpu_offload,
                        decoder_mse_threshold=args.decoder_mse_threshold,
                    )
                )
            slots = tuple(AsyncPersonaPlexSlot(engine) for engine in engines)
            # CUDA graph capture is process-global, so threaded startup must
            # serialize first capture even though the devices differ.
            for slot in slots:
                await slot.warmup()
            sample_rate = engines[0].mimi.sample_rate
            frame_size = engines[0].frame_size
        memory_worker = (
            HttpMemoryWorker(
                args.memory_worker_url,
                queue_frames=args.memory_queue_frames,
            )
            if args.memory_worker_url
            else NullMemoryWorker()
        )
        config = ControllerConfig(
            install_backend=args.install_backend,
            context_capacity_frames=args.context_capacity_frames,
            trigger_reserve_frames=args.trigger_reserve_frames,
            prefill_batch_frames=args.prefill_batch_frames,
            text_prefill_chunk_tokens=args.text_prefill_chunk_tokens,
            inplace_min_tail_frames=args.inplace_min_tail_frames,
            switch_max_lag_frames=args.switch_max_lag_frames,
            safe_silence_frames=args.safe_silence_frames,
            silence_rms_threshold=args.silence_rms_threshold,
            force_switch_after_s=args.force_switch_after_s,
            install_timeout_s=args.install_timeout_s,
            ledger_max_frames=args.ledger_max_frames,
            codec_convergence_frames=args.codec_convergence_frames,
            decoder_convergence_frames=args.decoder_convergence_frames,
            auto_trigger_enabled=not args.disable_auto_trigger,
        )
        controller = SeamlessInstallController(slots, memory_worker, config=config)
        state = SeamlessWebState(
            slots=slots,
            controller=controller,
            voice_prompt_dir=voice_prompt_dir,
            sample_rate=sample_rate,
            frame_size=frame_size,
            control_token=args.control_token,
        )
        app = web.Application()
        app.router.add_get("/api/chat", state.handle_chat)
        app.router.add_get("/api/seamless/status", state.handle_status)
        app.router.add_post("/api/seamless/install", state.handle_install)
        if static_path is not None:

            async def handle_root(request: web.Request) -> web.FileResponse:
                del request
                return web.FileResponse(os.path.join(static_path, "index.html"))

            app.router.add_get("/", handle_root)
            app.router.add_static("/", path=static_path, follow_symlinks=True)
        app.on_cleanup.append(state.cleanup)
        return app

    ssl_context = None
    protocol = "http"
    if args.ssl:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = (
        args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    )
    logger.info("WebDemo: %s://%s:%d", protocol, host_ip, args.port)
    if args.install_backend == InstallBackend.SHADOW.value:
        logger.info(
            "GPU routing: backend=shadow memory=%s active=%s shadow=%s",
            args.memory_worker_url or "manual",
            args.active_device,
            args.shadow_device,
        )
    else:
        logger.info(
            "GPU routing: backend=inplace memory=%s active=%s",
            args.memory_worker_url or "manual",
            args.active_device,
        )
    if args.control_token is None and args.host not in ("localhost", "127.0.0.1"):
        logger.warning(
            "manual install endpoint has no --control-token on a non-local host"
        )
    web.run_app(make_app(), host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()
