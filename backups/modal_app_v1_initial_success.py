import os
import time
import logging
import threading
import asyncio
from pathlib import Path
import modal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the Modal App
app = modal.App("ap-og")

# Define the container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "sentencepiece",
        "numpy",
        "aiohttp",
        "huggingface_hub",
        "safetensors",
        "sphn==0.1.12",
        "accelerate",
        "fastapi",
        "starlette",
        "uvicorn",
        "einops"
    )
    .env({"PYTHONPATH": "/root/moshi"})
    .add_local_dir(Path(__file__).parent / "moshi", remote_path="/root/moshi")
)

volume = modal.Volume.from_name("personaplex-weights", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/root/weights": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=300
)
class PersonaPlexServer:
    def __init__(self):
        self.state = None
        self.init_error = None
        self.progress = "Not started"
        self._load_lock = threading.Lock()

    def _do_load(self):
        with self._load_lock:
            if self.state is not None: return
            self.progress = "Starting imports..."
            logger.info("--- LAZY LOAD STARTED ---")
            try:
                from moshi.server import ServerState, torch_auto_device, seed_all, _get_voice_prompt_dir
                from moshi.models import loaders
                import sentencepiece
                import torch
                self.progress = "Initializing CUDA..."
                self.device = torch_auto_device("cuda")
                seed_all(42424242)
                repo = loaders.DEFAULT_REPO
                weights_dir = Path("/root/weights")
                tokenizer_path = weights_dir / loaders.TEXT_TOKENIZER_NAME
                mimi_path = weights_dir / loaders.MIMI_NAME
                moshi_path = weights_dir / loaders.MOSHI_NAME
                self.progress = "Loading Mimi..."
                mimi = loaders.get_mimi(mimi_path, self.device)
                other_mimi = loaders.get_mimi(mimi_path, self.device)
                self.progress = "Loading Tokenizer..."
                text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
                self.progress = "Loading Moshi (14GB)..."
                lm = loaders.get_moshi_lm(moshi_path, device=self.device)
                self.progress = "Warming up..."
                voice_dir = _get_voice_prompt_dir(None, repo)
                self.state = ServerState(mimi=mimi, other_mimi=other_mimi, text_tokenizer=text_tokenizer, lm=lm, device=self.device, voice_prompt_dir=voice_dir)
                self.state.warmup()
                self.progress = "Ready"
                logger.info("🚀 BACKEND READY!")
            except Exception as e:
                import traceback
                self.init_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                self.progress = "Failed"
                logger.error(f"❌ LOAD FAILED: {self.init_error}")

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        import asyncio
        import torch
        import numpy as np
        import sphn
        from moshi.server import wrap_with_system_tags, seed_all
        from starlette.websockets import WebSocketState

        fastapi_app = FastAPI()
        fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        @fastapi_app.get("/health")
        def health():
            if self.state is None and self.progress == "Not started": threading.Thread(target=self._do_load).start()
            return JSONResponse({"ready": self.state is not None, "progress": self.progress, "error": self.init_error})

        @fastapi_app.websocket("/api/chat")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            if self.state is None:
                await websocket.send_bytes(b"\x02" + f"Backend is loading: {self.progress}".encode("utf8"))
                await websocket.close(code=1001)
                return

            logger.info("New WebSocket connection")
            query = websocket.query_params
            voice_prompt_filename = query.get("voice_prompt")
            voice_prompt_path = os.path.join(self.state.voice_prompt_dir, voice_prompt_filename) if self.state.voice_prompt_dir and voice_prompt_filename else None
            
            close = False
            async def is_alive(): return not close

            try:
                async with self.state.lock:
                    if voice_prompt_path and os.path.exists(voice_prompt_path):
                        if self.state.lm_gen.voice_prompt != voice_prompt_path:
                            logger.info(f"Loading voice prompt: {voice_prompt_path}")
                            if voice_prompt_path.endswith('.pt'):
                                self.state.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
                            else:
                                self.state.lm_gen.load_voice_prompt(voice_prompt_path)
                    
                    text_prompt = query.get("text_prompt", "")
                    self.state.lm_gen.text_prompt_tokens = self.state.text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
                    seed = int(query.get("audio_seed", query.get("text_seed", query.get("seed", -1))))
                    if seed != -1: seed_all(seed)

                    self.state.mimi.reset_streaming()
                    self.state.other_mimi.reset_streaming()
                    self.state.lm_gen.reset_streaming()
                    
                    await self.state.lm_gen.step_system_prompts_async(self.state.mimi, is_alive=is_alive)
                    self.state.mimi.reset_streaming()
                    await websocket.send_bytes(b"\x00")
                    logger.info("Handshake sent.")

                    opus_writer = sphn.OpusStreamWriter(self.state.mimi.sample_rate)
                    opus_reader = sphn.OpusStreamReader(self.state.mimi.sample_rate)

                    async def recv_loop():
                        nonlocal close
                        try:
                            while not close:
                                msg = await websocket.receive()
                                if msg["type"] == "websocket.disconnect": break
                                elif "bytes" in msg:
                                    data = msg["bytes"]
                                    if len(data) > 0:
                                        if data[0] == 1: opus_reader.append_bytes(data[1:])
                        except: pass
                        finally: close = True

                    async def send_loop():
                        try:
                            while not close:
                                await asyncio.sleep(0.01)
                                msg = opus_writer.read_bytes()
                                if len(msg) > 0: await websocket.send_bytes(b"\x01" + msg)
                        except: pass
                        finally: close = True

                    async def opus_loop():
                        nonlocal close
                        try:
                            all_pcm_data = None
                            while not close:
                                await asyncio.sleep(0.005)
                                pcm = opus_reader.read_pcm()
                                if pcm is not None and pcm.shape[-1] > 0:
                                    all_pcm_data = pcm if all_pcm_data is None else np.concatenate((all_pcm_data, pcm))
                                    while all_pcm_data is not None and all_pcm_data.shape[-1] >= self.state.frame_size:
                                        chunk = torch.from_numpy(all_pcm_data[:self.state.frame_size]).to(device=self.state.device, dtype=torch.float32)[None, None]
                                        all_pcm_data = all_pcm_data[self.state.frame_size:]
                                        with torch.no_grad():
                                            codes = self.state.mimi.encode(chunk)
                                            _ = self.state.other_mimi.encode(chunk)
                                            for c in range(codes.shape[-1]):
                                                tokens = self.state.lm_gen.step(codes[:, :, c: c + 1])
                                                if tokens is None: continue
                                                main_pcm = self.state.mimi.decode(tokens[:, 1:9])
                                                _ = self.state.other_mimi.decode(tokens[:, 1:9])
                                                opus_writer.append_pcm(main_pcm[0, 0].detach().cpu().numpy())
                                                text_token = tokens[0, 0, 0].item()
                                                if text_token not in (0, 3):
                                                    text = self.state.text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                                                    await websocket.send_bytes(b"\x02" + text.encode("utf8"))
                        except: pass
                        finally: close = True

                    tasks = [asyncio.create_task(recv_loop()), asyncio.create_task(send_loop()), asyncio.create_task(opus_loop())]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending: t.cancel()
            except WebSocketDisconnect: pass
            finally:
                try: await websocket.close()
                except: pass

        return fastapi_app

@app.function(image=image, volumes={"/root/weights": volume}, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def download_models():
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    hf_token = os.environ.get("HF_TOKEN")
    weights_dir = Path("/root/weights"); weights_dir.mkdir(exist_ok=True, parents=True)
    for f in [loaders.TEXT_TOKENIZER_NAME, loaders.MIMI_NAME, loaders.MOSHI_NAME]:
        hf_hub_download(loaders.DEFAULT_REPO, f, token=hf_token, local_dir=weights_dir, local_dir_use_symlinks=False)
    volume.commit()
