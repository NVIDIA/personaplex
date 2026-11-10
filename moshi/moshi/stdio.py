#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


"""
Real-time stdin/stdout streaming entrypoint for PersonaPlex.

Protocol:
- Transport frame: [u32 little-endian payload_len][payload bytes]
- payload[0] is the message kind:
    0x00 handshake (stdout)
    0x01 audio (stdin/stdout), PCM16LE mono at model sample rate
    0x02 text  (stdout), UTF-8
    0x05 error (stdout), UTF-8
    0x06 ping  (stdin), ignored
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import struct
import sys
import tarfile
from dataclasses import dataclass
from typing import BinaryIO, Optional

from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import torch

from .client_utils import make_log
from .models import loaders, LMGen, MimiModel

KIND_HANDSHAKE = 0x00
KIND_AUDIO = 0x01
KIND_TEXT = 0x02
KIND_ERROR = 0x05
KIND_PING = 0x06


def log(level: str, msg: str):
    print(make_log(level, msg), file=sys.stderr, flush=True)


def seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    log("info", "retrieving voice prompts")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        log("info", f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")
    return str(voices_dir)


def warmup(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, device: torch.device, frame_size: int):
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        with contextlib.redirect_stdout(sys.stderr):
            codes = mimi.encode(chunk)
            _ = other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
    if device.type == "cuda":
        torch.cuda.synchronize()


def pcm16le_bytes_to_float32(pcm16le: bytes) -> np.ndarray:
    if len(pcm16le) % 2 != 0:
        raise ValueError("PCM16 payload must contain an even number of bytes.")
    pcm_i16 = np.frombuffer(pcm16le, dtype="<i2")
    return pcm_i16.astype(np.float32) / 32768.0


def float32_to_pcm16le_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm_i16 = np.round(clipped * 32767.0).astype("<i2")
    return pcm_i16.tobytes()


def encode_packet(kind: int, payload: bytes = b"") -> bytes:
    if kind < 0 or kind > 255:
        raise ValueError(f"Invalid packet kind: {kind}")
    frame_payload = bytes([kind]) + payload
    return struct.pack("<I", len(frame_payload)) + frame_payload


class LengthPrefixedParser:
    def __init__(self, max_payload_bytes: int):
        self.max_payload_bytes = max_payload_bytes
        self._buffer = bytearray()

    @property
    def has_pending_data(self) -> bool:
        return len(self._buffer) > 0

    def feed(self, data: bytes) -> list[bytes]:
        if data:
            self._buffer.extend(data)

        packets: list[bytes] = []
        while True:
            if len(self._buffer) < 4:
                break
            payload_len = struct.unpack_from("<I", self._buffer, 0)[0]
            if payload_len == 0:
                raise ValueError("Invalid zero-length payload.")
            if payload_len > self.max_payload_bytes:
                raise ValueError(
                    f"Payload length {payload_len} exceeds max_payload_bytes={self.max_payload_bytes}."
                )
            end_idx = 4 + payload_len
            if len(self._buffer) < end_idx:
                break
            packets.append(bytes(self._buffer[4:end_idx]))
            del self._buffer[:end_idx]
        return packets


class PacketWriter:
    def __init__(self, stream: BinaryIO, max_payload_bytes: int):
        self.stream = stream
        self.max_payload_bytes = max_payload_bytes
        self.closed = False

    def send(self, kind: int, payload: bytes = b"") -> bool:
        if self.closed:
            return False
        if len(payload) + 1 > self.max_payload_bytes:
            raise ValueError(
                f"payload for kind={kind} exceeds max_payload_bytes={self.max_payload_bytes}: "
                f"{len(payload) + 1} bytes"
            )
        packet = encode_packet(kind, payload)
        try:
            self.stream.write(packet)
            self.stream.flush()
            return True
        except (BrokenPipeError, OSError):
            self.closed = True
            log("warning", "stdout pipe closed; stopping.")
            return False

    def send_error(self, message: str) -> bool:
        return self.send(KIND_ERROR, message.encode("utf-8", errors="replace"))


class PCMFrameBuffer:
    """Buffers PCM16 bytes and emits exact model-sized float32 frames."""

    def __init__(self, frame_size_samples: int):
        self.frame_size_samples = frame_size_samples
        self.frame_size_bytes = frame_size_samples * 2
        self._buffer = bytearray()

    def append_pcm16(self, payload: bytes):
        if len(payload) % 2 != 0:
            raise ValueError("audio payload length must be an even number of bytes.")
        self._buffer.extend(payload)

    def pop_complete_frames(self) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        while len(self._buffer) >= self.frame_size_bytes:
            frame_bytes = bytes(self._buffer[: self.frame_size_bytes])
            del self._buffer[: self.frame_size_bytes]
            frames.append(pcm16le_bytes_to_float32(frame_bytes))
        return frames

    def flush_padded_frame(self) -> Optional[np.ndarray]:
        if len(self._buffer) == 0:
            return None
        if len(self._buffer) % 2 != 0:
            # Should not happen if append_pcm16 validated, but keep this safe.
            self._buffer = self._buffer[:-1]
        padded = bytes(self._buffer) + b"\x00" * (self.frame_size_bytes - len(self._buffer))
        self._buffer.clear()
        return pcm16le_bytes_to_float32(padded)


@dataclass
class RuntimeState:
    mimi: MimiModel
    other_mimi: MimiModel
    lm_gen: LMGen
    text_tokenizer: sentencepiece.SentencePieceProcessor
    device: torch.device
    frame_size: int


def _init_runtime(
    voice_prompt_path: str,
    text_prompt: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: torch.device,
    seed: int,
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    cpu_offload: bool,
    save_voice_prompt_embeddings: bool,
) -> RuntimeState:
    if seed != -1:
        seed_all(seed)

    hf_hub_download(hf_repo, "config.json")

    log("info", "loading mimi")
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)  # type: ignore
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    log("info", "mimi loaded")

    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)  # type: ignore
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore

    log("info", "loading moshi")
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
    lm.eval()
    log("info", "moshi loaded")

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        use_sampling=not greedy,
        temp=temp_audio,
        temp_text=temp_text,
        top_k=topk_audio,
        top_k_text=topk_text,
    )

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    log("info", "warming up model")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    if voice_prompt_path.endswith(".pt"):
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
    )

    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()
    with contextlib.redirect_stdout(sys.stderr):
        lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()
    log("info", "done with system prompts")

    return RuntimeState(
        mimi=mimi,
        other_mimi=other_mimi,
        lm_gen=lm_gen,
        text_tokenizer=text_tokenizer,
        device=device,
        frame_size=frame_size,
    )


def _emit_model_step_output(state: RuntimeState, writer: PacketWriter, frame: np.ndarray) -> bool:
    chunk = torch.from_numpy(frame).to(device=state.device)[None, None]
    with contextlib.redirect_stdout(sys.stderr):
        codes = state.mimi.encode(chunk)
        _ = state.other_mimi.encode(chunk)

        for c in range(codes.shape[-1]):
            tokens = state.lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
            decoded = state.mimi.decode(tokens[:, 1:9])
            _ = state.other_mimi.decode(tokens[:, 1:9])

            pcm = decoded.detach().cpu().numpy()[0, 0]
            if not writer.send(KIND_AUDIO, float32_to_pcm16le_bytes(pcm)):
                return False

            text_token = tokens[0, 0, 0].item()
            if text_token not in (0, 3):
                text_piece = state.text_tokenizer.id_to_piece(text_token)  # type: ignore
                text_piece = text_piece.replace("▁", " ")
                if not writer.send(KIND_TEXT, text_piece.encode("utf-8")):
                    return False
    return True


def run_stdio_stream(
    state: RuntimeState,
    writer: PacketWriter,
    stdin_stream: BinaryIO,
    read_size: int,
    max_payload_bytes: int,
    eof_drain_frames: int,
) -> int:
    parser = LengthPrefixedParser(max_payload_bytes=max_payload_bytes)
    pcm_buffer = PCMFrameBuffer(frame_size_samples=state.frame_size)

    if not writer.send(KIND_HANDSHAKE):
        return 1

    while True:
        chunk = stdin_stream.read(read_size)
        if not chunk:
            break
        try:
            messages = parser.feed(chunk)
        except ValueError as exc:
            err_msg = f"protocol parse error: {exc}"
            log("error", err_msg)
            writer.send_error(err_msg)
            return 2

        for message in messages:
            if not message:
                msg = "received empty payload."
                log("warning", msg)
                writer.send_error(msg)
                continue
            kind = message[0]
            payload = message[1:]
            if kind == KIND_AUDIO:
                try:
                    pcm_buffer.append_pcm16(payload)
                except ValueError as exc:
                    err_msg = f"invalid audio payload: {exc}"
                    log("warning", err_msg)
                    writer.send_error(err_msg)
                    continue
                for frame in pcm_buffer.pop_complete_frames():
                    if not _emit_model_step_output(state, writer, frame):
                        return 0
            elif kind == KIND_PING:
                continue
            else:
                msg = f"unknown message kind {kind}"
                log("warning", msg)
                writer.send_error(msg)

    if parser.has_pending_data:
        msg = "stdin ended with incomplete frame payload."
        log("warning", msg)
        writer.send_error(msg)

    final_frame = pcm_buffer.flush_padded_frame()
    if final_frame is not None:
        if not _emit_model_step_output(state, writer, final_frame):
            return 0

    if eof_drain_frames > 0:
        log("info", f"draining EOF with {eof_drain_frames} silence frames")
        zero_frame = np.zeros((state.frame_size,), dtype=np.float32)
        for _ in range(eof_drain_frames):
            if not _emit_model_step_output(state, writer, zero_frame):
                return 0
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Real-time stdin/stdout streaming for PersonaPlex. "
            "Input and output use [u32 payload_len][payload] framing."
        )
    )
    parser.add_argument(
        "--voice-prompt",
        required=True,
        type=str,
        help="Voice prompt filename (basename) inside --voice-prompt-dir (e.g. 'NATM1.pt').",
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
        ),
    )
    parser.add_argument(
        "--text-prompt",
        default=(
            "You are a wise and friendly teacher. "
            "Answer questions or provide advice in a clear and engaging way."
        ),
        type=str,
        help="Text prompt.",
    )
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into (defaults to pre-trained model repo).",
    )
    parser.add_argument("--temp-audio", type=float, default=0.8, help="Audio sampling temperature.")
    parser.add_argument("--temp-text", type=float, default=0.7, help="Text sampling temperature.")
    parser.add_argument("--topk-audio", type=int, default=250, help="Audio top-k sampling.")
    parser.add_argument("--topk-text", type=int, default=25, help="Text top-k sampling.")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling (greedy decoding).")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload LM model layers to CPU when GPU memory is insufficient.",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Seed for reproducibility (-1 disables).")
    parser.add_argument(
        "--read-size",
        type=int,
        default=4096,
        help="Number of bytes per stdin read iteration.",
    )
    parser.add_argument(
        "--max-payload-bytes",
        type=int,
        default=8 * 1024 * 1024,
        help="Maximum payload bytes for any framed message (after u32 length).",
    )
    parser.add_argument(
        "--eof-drain-frames",
        type=int,
        default=32,
        help=(
            "Extra silence frames to process after stdin EOF. "
            "Default: 32 (~2.56s at 12.5fps). Use -1 for auto (LM max delay), 0 to disable."
        ),
    )
    args = parser.parse_args()

    voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if not os.path.exists(voice_prompt_dir):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir}")
    voice_prompt_path = os.path.join(voice_prompt_dir, args.voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(
            f"Voice prompt '{args.voice_prompt}' not found in "
            f"'{voice_prompt_dir}' (resolved: {voice_prompt_path})"
        )

    wire_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr
    writer = PacketWriter(stream=wire_stdout, max_payload_bytes=args.max_payload_bytes)

    device = torch.device(args.device)
    with torch.no_grad():
        state = _init_runtime(
            voice_prompt_path=voice_prompt_path,
            text_prompt=args.text_prompt,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            hf_repo=args.hf_repo,
            device=device,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=bool(args.greedy),
            cpu_offload=args.cpu_offload,
            save_voice_prompt_embeddings=False,
        )
        eof_drain_frames = (
            int(state.lm_gen.max_delay) if int(args.eof_drain_frames) < 0 else int(args.eof_drain_frames)
        )
        code = run_stdio_stream(
            state=state,
            writer=writer,
            stdin_stream=sys.stdin.buffer,
            read_size=args.read_size,
            max_payload_bytes=args.max_payload_bytes,
            eof_drain_frames=eof_drain_frames,
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
