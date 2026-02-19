#!/usr/bin/env python3
"""
Concise check for moshi.stdio using assets in assets/test.

Default behavior: validate framing roundtrip (no model run).
Runs end-to-end moshi.stdio by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import wave

KIND_AUDIO = 0x01


def encode_packet(kind: int, payload: bytes = b"") -> bytes:
    frame_payload = bytes([kind]) + payload
    return struct.pack("<I", len(frame_payload)) + frame_payload


def decode_packets(data: bytes, max_payload_bytes: int = 32 * 1024 * 1024) -> list[bytes]:
    out: list[bytes] = []
    offset = 0
    n = len(data)
    while True:
        if offset + 4 > n:
            if offset == n:
                return out
            raise ValueError(f"incomplete frame header at offset {offset}")
        payload_len = struct.unpack_from("<I", data, offset)[0]
        if payload_len <= 0:
            raise ValueError(f"invalid payload_len={payload_len} at offset {offset}")
        if payload_len > max_payload_bytes:
            raise ValueError(f"payload_len={payload_len} exceeds {max_payload_bytes} at offset {offset}")
        offset += 4
        if offset + payload_len > n:
            raise ValueError(f"incomplete frame payload at offset {offset - 4}")
        out.append(data[offset : offset + payload_len])
        offset += payload_len


def read_wav_pcm16_mono(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())
    if channels != 1:
        raise ValueError(f"expected mono wav; got channels={channels}")
    if sample_width != 2:
        raise ValueError(f"expected 16-bit wav; got sample_width={sample_width}")
    return pcm_bytes, sample_rate


def write_wav_pcm16_mono(path: Path, pcm_bytes: bytes, sample_rate: int):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def framing_roundtrip(pcm_bytes: bytes, sample_rate: int, chunk_bytes: int, out_wav: Path | None):
    framed = bytearray()
    for i in range(0, len(pcm_bytes), chunk_bytes):
        framed.extend(encode_packet(KIND_AUDIO, pcm_bytes[i : i + chunk_bytes]))

    out_audio = bytearray()
    for payload in decode_packets(bytes(framed)):
        if payload[0] != KIND_AUDIO:
            continue
        out_audio.extend(payload[1:])

    if bytes(out_audio) != pcm_bytes:
        raise AssertionError("framing roundtrip mismatch")
    if out_wav is not None:
        write_wav_pcm16_mono(out_wav, bytes(out_audio), sample_rate)


def run_e2e(
    pcm_bytes: bytes,
    chunk_bytes: int,
    python_bin: str,
    moshi_args: list[str],
    output_wav: Path,
    output_text: Path | None,
):
    framed_in = bytearray()
    for i in range(0, len(pcm_bytes), chunk_bytes):
        framed_in.extend(encode_packet(KIND_AUDIO, pcm_bytes[i : i + chunk_bytes]))

    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])  # .../personaplex/moshi
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{package_root}:{existing_pythonpath}" if existing_pythonpath else package_root

    cmd = [python_bin, "-m", "moshi.stdio"] + moshi_args
    print("running:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(
        cmd,
        input=bytes(framed_in),
        stdout=subprocess.PIPE,
        stderr=None,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"moshi.stdio exited with code {proc.returncode}")

    packets = decode_packets(proc.stdout)
    audio_out = bytearray()
    text_out: list[str] = []
    for payload in packets:
        kind = payload[0]
        body = payload[1:]
        if kind == KIND_AUDIO:
            audio_out.extend(body)
        elif kind == 0x02:
            text_out.append(body.decode("utf-8", errors="replace"))

    write_wav_pcm16_mono(output_wav, bytes(audio_out), 24000)
    if output_text is not None:
        output_text.write_text("".join(text_out), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test runner for moshi.stdio.")
    parser.add_argument(
        "--input-wav",
        default=str(Path(__file__).resolve().parents[2] / "assets/test/input_assistant.wav"),
        help="Input WAV (mono PCM16).",
    )
    parser.add_argument("--chunk-bytes", type=int, default=1000, help="PCM bytes per audio packet.")
    parser.add_argument("--out-wav", help="Optional output wav for framing roundtrip.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for subprocess.")
    parser.add_argument(
        "--moshi-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Args forwarded to `python -m moshi.stdio`.",
    )
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--e2e-out-wav",
        default=str(repo_root / "moshi_stdio_out.wav"),
        help="Output wav for e2e.",
    )
    parser.add_argument(
        "--e2e-out-text",
        default=str(repo_root / "moshi_stdio_out.txt"),
        help="Optional output text for e2e.",
    )
    args = parser.parse_args()

    pcm_bytes, sample_rate = read_wav_pcm16_mono(Path(args.input_wav))
    out_wav = Path(args.out_wav) if args.out_wav else None
    framing_roundtrip(pcm_bytes, sample_rate, args.chunk_bytes, out_wav)

    if not args.moshi_args:
        raise SystemExit(
            "moshi args required. Example: "
            "--moshi-args --voice-prompt NATM1.pt --device cuda"
        )

    output_wav = Path(args.e2e_out_wav)
    output_text = Path(args.e2e_out_text) if args.e2e_out_text else None
    run_e2e(
        pcm_bytes=pcm_bytes,
        chunk_bytes=args.chunk_bytes,
        python_bin=args.python_bin,
        moshi_args=args.moshi_args,
        output_wav=output_wav,
        output_text=output_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
