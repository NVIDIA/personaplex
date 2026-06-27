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


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import numpy as np
import sentencepiece
import sphn
import torch
import random

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)
DeviceString = Literal["auto", "cuda", "cpu", "mps"]
MOSHI_MLX_NATIVE_NAME = "model.moshi_mlx.safetensors"

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested not in (None, "auto"):
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    if str(path).lower() in {"", "none"}:
        return None
    expanded = Path(path).expanduser()
    return expanded if expanded.exists() else None


def _find_local_file(root: Path | None, *names: str) -> Path | None:
    if root is None:
        return None
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _resolve_voice_prompt_path(voice_prompt_dir: str, requested: str | None) -> Path:
    if requested is None or requested == "":
        raise FileNotFoundError(f"No voice prompt requested from '{voice_prompt_dir}'")

    requested_path = Path(requested)
    if requested_path.name != requested:
        raise FileNotFoundError(f"Invalid voice prompt name '{requested}'")

    root = Path(voice_prompt_dir)
    candidate = root / requested
    if candidate.exists():
        return candidate

    stem = requested_path.stem
    for suffix in (".safetensors", ".pt", ".pth", ".wav", ".mp3", ".flac", ".ogg"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Requested voice prompt '{requested}' not found in '{voice_prompt_dir}'")


def _is_voice_prompt_embeddings(path: str | Path) -> bool:
    return Path(path).suffix in {".safetensors", ".pt", ".pth"}


def _browser_host(host: str) -> str:
    if host in ("0.0.0.0", "::", "", "localhost"):
        return "localhost"
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _log_server_urls(protocol: str, host: str, port: int) -> None:
    local_url = f"{protocol}://{_browser_host(host)}:{port}"
    logger.info(f"Serving Web UI at {local_url}")
    if host in ("0.0.0.0", "::", ""):
        lan_ip = get_lan_ip()
        if lan_ip not in ("127.0.0.1", "localhost"):
            logger.info(f"Serving Web UI on LAN at {protocol}://{lan_ip}:{port}")


def _download_or_explain(hf_repo: str, filename: str, purpose: str) -> str:
    try:
        return hf_hub_download(hf_repo, filename)
    except HfHubHTTPError:
        raise SystemExit(
            f"Could not download {purpose} ({filename}) from {hf_repo}. "
            "If you want Hugging Face downloads, accept the model license and export HF_TOKEN. "
            "For offline/local startup, pass the local file explicitly or place it under --model-dir."
        ) from None


def _apply_local_model_defaults(args) -> Path | None:
    model_dir = _optional_path(args.model_dir)
    if model_dir is None:
        return None

    if args.voice_prompt_dir is None and (model_dir / "voices").exists():
        args.voice_prompt_dir = str(model_dir / "voices")
    if args.moshi_weight is None:
        local_moshi = _find_local_file(model_dir, loaders.MOSHI_NAME, "model.safetensors")
        if local_moshi is not None:
            args.moshi_weight = str(local_moshi)
    if args.mimi_weight is None:
        local_mimi = _find_local_file(model_dir, loaders.MIMI_NAME, "tokenizer.safetensors")
        if local_mimi is not None:
            args.mimi_weight = str(local_mimi)
    if args.tokenizer is None:
        local_tokenizer = _find_local_file(model_dir, loaders.TEXT_TOKENIZER_NAME)
        if local_tokenizer is not None:
            args.tokenizer = str(local_tokenizer)

    if args.static is None:
        for candidate in (model_dir / "dist", Path("client/dist")):
            if candidate.exists():
                args.static = str(candidate)
                break
        else:
            args.static = "none"
            logger.info("no local static UI found; serving API only. Build client/dist or pass --static to serve UI.")

    return model_dir


def _require_existing_path(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing {purpose}: {path}")
    return path


def _run_mlx_server(args, static_path: str | None) -> None:
    mlx_dir = _require_existing_path(Path(args.mlx_model_dir).expanduser(), "MLX model directory")
    moshi_weight = Path(args.moshi_weight).expanduser() if args.moshi_weight else mlx_dir / MOSHI_MLX_NATIVE_NAME
    mimi_weight = Path(args.mimi_weight).expanduser() if args.mimi_weight else (
        _find_local_file(mlx_dir, "tokenizer.safetensors", loaders.MIMI_NAME) or mlx_dir / "tokenizer.safetensors"
    )
    if args.tokenizer:
        tokenizer = Path(args.tokenizer).expanduser()
    else:
        tokenizer = (
            _find_local_file(mlx_dir, loaders.TEXT_TOKENIZER_NAME)
            or _find_local_file(_optional_path(args.model_dir), loaders.TEXT_TOKENIZER_NAME)
        )
        if tokenizer is None:
            tokenizer = Path(_download_or_explain(args.hf_repo, loaders.TEXT_TOKENIZER_NAME, "text tokenizer"))
    lm_config = mlx_dir / "config.json"

    if not moshi_weight.exists() and (mlx_dir / loaders.MOSHI_NAME).exists():
        raise SystemExit(
            f"Missing MLX-native Moshi weights: {moshi_weight}\n"
            "Create them with:\n"
            f"  python -m moshi.convert_mlx --input {args.model_dir} --output-dir {args.mlx_model_dir} "
            "--force --moshi-mlx-native on"
        )

    _require_existing_path(moshi_weight, "MLX-native Moshi weights")
    _require_existing_path(mimi_weight, "Mimi tokenizer")
    _require_existing_path(tokenizer, "text tokenizer")
    _require_existing_path(lm_config, "MLX model config")

    try:
        from moshi_mlx import local_web as mlx_local_web
    except Exception as exc:
        raise SystemExit(
            "moshi_mlx is required for --device mps. "
            "Run ./scripts/bootstrap-nix-macos.sh to install the MLX runtime."
        ) from exc

    host = "localhost" if args.host in ("0.0.0.0", "::", "") else args.host
    argv = [
        "moshi_mlx.local_web",
        "--moshi-weight", str(moshi_weight),
        "--mimi-weight", str(mimi_weight),
        "--tokenizer", str(tokenizer),
        "--lm-config", str(lm_config),
        "--host", host,
        "--port", str(args.port),
        "--no-browser",
    ]
    if static_path is not None:
        argv.extend(["--static", static_path])
    else:
        argv.extend(["--static", "none"])
    if args.ssl is not None:
        create_ssl_context(args.ssl)
        argv.extend(["--ssl", args.ssl])

    logger.info("device=mps selected; launching moshi_mlx runtime")
    logger.info(f"MLX model_dir = {mlx_dir}")
    original_argv = sys.argv
    try:
        sys.argv = argv
        mlx_local_web.main()
    finally:
        sys.argv = original_argv


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, other_mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False, prompt_prefill: bool = True):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.prompt_prefill = prompt_prefill
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
    
    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()


    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # self.lm_gen.temp = float(request.query["audio_temperature"])
        # self.lm_gen.temp_text = float(request.query["text_temperature"])
        # self.lm_gen.top_k_text = max(1, int(request.query["text_topk"]))
        # self.lm_gen.top_k = max(1, int(request.query["audio_topk"]))
        
        text_prompt = request.query.get("text_prompt", "")
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query.get("voice_prompt")
            voice_prompt_path = str(_resolve_voice_prompt_path(self.voice_prompt_dir, voice_prompt_filename))

        if voice_prompt_path is not None and self.lm_gen.voice_prompt != voice_prompt_path:
            if _is_voice_prompt_embeddings(voice_prompt_path):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
            if len(text_prompt) > 0 else None
        )
        seed = int(request.query["seed"]) if "seed" in request.query else None

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if len(text_prompt) > 0:
            clog.log("info", f"text prompt: {text_prompt}")
        if voice_prompt_path is not None:
            clog.log("info", f"voice prompt: {voice_prompt_path}")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            async def is_alive():
                return not close and not ws.closed

            if self.prompt_prefill:
                # Reuse mimi for encoding voice prompt and then reset it before conversation starts.
                prompt_start = time.time()
                clog.log("info", "loading system prompts")
                await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
                self.mimi.reset_streaming()
                clog.log("info", f"done with system prompts in {time.time() - prompt_start:.2f}s")
            else:
                clog.log("warning", "skipping system prompt prefill for this backend")
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
                # await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = _download_or_explain(hf_repo, "voices.tgz", "voice prompts")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str], hf_repo: str) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = _download_or_explain(hf_repo, "dist.tgz", "static Web UI")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--model-dir", type=str, default="model",
                        help="Local model directory to use before Hugging Face downloads. Defaults to ./model.")
    parser.add_argument("--mlx-model-dir", type=str, default="model-mlx",
                        help="Local MLX model directory used when --device resolves to mps. Defaults to ./model-mlx.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device on which to run: auto, cuda, mps, or cpu. Defaults to auto.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument("--prompt-prefill", choices=("auto", "on", "off"), default="auto",
                        help="Apply voice/text prompts before accepting audio. "
                             "Defaults to on for CUDA and off for MPS/CPU.")
    parser.add_argument("--warmup", choices=("auto", "on", "off"), default="auto",
                        help="Run model warmup before serving. Defaults to on for CUDA and off for MPS/CPU.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    explicit_moshi_weight = args.moshi_weight is not None
    explicit_mimi_weight = args.mimi_weight is not None
    explicit_tokenizer = args.tokenizer is not None
    model_dir = _apply_local_model_defaults(args)
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static, args.hf_repo)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    logger.info(f"device = {args.device}")
    if args.device.type == "mps":
        if not explicit_moshi_weight:
            args.moshi_weight = None
        if not explicit_mimi_weight:
            args.mimi_weight = None
        if not explicit_tokenizer:
            args.tokenizer = None
        _run_mlx_server(args, static_path)
        return
    prompt_prefill = (
        args.prompt_prefill == "on"
        or (args.prompt_prefill == "auto" and args.device.type == "cuda")
    )
    if not prompt_prefill and args.prompt_prefill == "auto":
        logger.warning(
            f"prompt prefill disabled on {args.device.type}; "
            "use --prompt-prefill on to force voice/text prompt conditioning."
        )
    logger.info(f"prompt_prefill = {prompt_prefill}")
    run_warmup = args.warmup == "on" or (args.warmup == "auto" and args.device.type == "cuda")
    if not run_warmup and args.warmup == "auto":
        logger.warning(f"warmup disabled on {args.device.type}; use --warmup on to force it.")
    logger.info(f"warmup = {run_warmup}")
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    if _find_local_file(model_dir, "config.json") is None:
        # Download config.json to increment download counter. Skip this in
        # local model-dir mode because gated repos require authentication.
        _download_or_explain(args.hf_repo, "config.json", "model config")

    if args.mimi_weight is None:
        args.mimi_weight = _download_or_explain(args.hf_repo, loaders.MIMI_NAME, "Mimi weights")
    if args.tokenizer is None:
        args.tokenizer = _download_or_explain(args.hf_repo, loaders.TEXT_TOKENIZER_NAME, "text tokenizer")
    if args.moshi_weight is None:
        args.moshi_weight = _download_or_explain(args.hf_repo, loaders.MOSHI_NAME, "Moshi LM weights")

    logger.info("loading mimi")
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        prompt_prefill=prompt_prefill,
    )
    if run_warmup:
        logger.info("warming up the model")
        state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    _log_server_urls(protocol, args.host, args.port)
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context, print=None)

if __name__ == "__main__":
    main()
