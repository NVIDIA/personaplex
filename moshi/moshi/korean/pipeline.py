# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import logging
from typing import Optional

import numpy as np
import sphn
from aiohttp import web
import aiohttp

from .asr import KoreanASR
from .llm import KoreanLLM
from .tts import KoreanTTS
from .voices import DEFAULT_KOREAN_VOICE
from ..utils.logging import ColorizedLog

logger = logging.getLogger(__name__)

# PersonaPlex native sample rate
SAMPLE_RATE = 24000
# Frame size matching Moshi's frame size at 24kHz
FRAME_SIZE = 1920


class KoreanPipeline:
    """Orchestrates Korean ASR -> LLM -> TTS pipeline over WebSocket."""

    def __init__(
        self,
        asr: KoreanASR,
        llm: KoreanLLM,
        tts: KoreanTTS,
    ):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.lock = asyncio.Lock()
        logger.info("KoreanPipeline initialized")

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket handler for Korean conversation, matching the Moshi protocol."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote
        peer_port = request.transport.get_extra_info("peername")[1]
        clog.log("info", f"[KO] Incoming connection from {peer}:{peer_port}")

        text_prompt = request.query.get("text_prompt", "")
        voice_prompt = request.query.get("voice_prompt", DEFAULT_KOREAN_VOICE)

        close = False
        opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)

        # Pending text from ASR that needs LLM processing
        pending_text: Optional[str] = None
        pending_text_event = asyncio.Event()

        async def recv_loop():
            """Receive audio from client and run ASR."""
            nonlocal close, pending_text
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                    ):
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue

                    data = message.data
                    if not isinstance(data, bytes) or len(data) == 0:
                        continue

                    kind = data[0]
                    if kind == 1:  # audio
                        opus_reader.append_bytes(data[1:])
                        pcm = opus_reader.read_pcm()
                        if pcm.shape[-1] == 0:
                            continue

                        transcription = self.asr.add_audio(pcm)
                        if transcription:
                            clog.log("info", f"[KO] ASR: {transcription}")
                            # Send transcribed text back to client
                            text_msg = b"\x02" + bytes(
                                f"[User] {transcription}", encoding="utf8"
                            )
                            await ws.send_bytes(text_msg)
                            pending_text = transcription
                            pending_text_event.set()
                    elif kind == 3:  # control
                        if len(data) > 1 and data[1] == 0x03:  # restart
                            self.asr.reset()
                            self.llm.reset()
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "[KO] recv_loop closed")

        async def process_loop():
            """Process ASR text through LLM and TTS."""
            nonlocal close, pending_text
            while not close:
                await pending_text_event.wait()
                pending_text_event.clear()

                if close or pending_text is None:
                    continue

                user_text = pending_text
                pending_text = None

                try:
                    # Stream LLM response
                    llm_text_buffer = []
                    async for text_chunk in self.llm.generate(
                        user_text, system_prompt=text_prompt or None
                    ):
                        if close:
                            break
                        llm_text_buffer.append(text_chunk)

                        # Send text to client as it streams
                        text_msg = b"\x02" + bytes(text_chunk, encoding="utf8")
                        await ws.send_bytes(text_msg)

                    if close:
                        continue

                    # Synthesize full LLM response
                    full_response = "".join(llm_text_buffer)
                    if not full_response.strip():
                        continue

                    clog.log("info", f"[KO] LLM: {full_response[:80]}...")

                    # TTS: convert response to audio and stream it
                    for audio_chunk in self.tts.synthesize(full_response, voice_prompt):
                        if close:
                            break
                        opus_writer.append_pcm(audio_chunk)

                except Exception as e:
                    clog.log("error", f"[KO] process error: {e}")
                    error_msg = b"\x05" + bytes(str(e), encoding="utf8")
                    try:
                        await ws.send_bytes(error_msg)
                    except Exception:
                        pass

        async def send_loop():
            """Send TTS audio back to client."""
            while not close:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    try:
                        await ws.send_bytes(b"\x01" + msg)
                    except Exception:
                        break

        async with self.lock:
            self.asr.reset()
            self.llm.reset()

            # Send handshake
            await ws.send_bytes(b"\x00")
            clog.log("info", "[KO] sent handshake")

            tasks = [
                asyncio.create_task(recv_loop()),
                asyncio.create_task(process_loop()),
                asyncio.create_task(send_loop()),
            ]

            done, pending_tasks = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            close = True
            pending_text_event.set()  # unblock process_loop

            for task in pending_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await ws.close()
            clog.log("info", "[KO] session closed")

        return ws
