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

# Image definition
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch", "sentencepiece", "numpy", "aiohttp", 
        "huggingface_hub", "safetensors", "sphn==0.1.12", 
        "accelerate", "fastapi", "starlette", "uvicorn", "einops", "pyloudnorm"
    )
    .add_local_dir(Path(__file__).parent / "moshi", remote_path="/root/moshi", copy=True)
    .add_local_file(Path(__file__).parent / "pepper.wav", remote_path="/root/pepper.wav", copy=True)
    .run_commands("pip install -e /root/moshi")
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
        self.progress = "Not started"
        self._load_lock = threading.Lock()

    def _do_load(self):
        with self._load_lock:
            if self.state is not None: return
            self.progress = "Loading..."
            try:
                from moshi.server import ServerState, torch_auto_device, seed_all, _get_voice_prompt_dir
                from moshi.models import loaders
                import sentencepiece
                import torch
                
                device = torch_auto_device("cuda")
                seed_all(42424242)
                
                weights_dir = Path("/root/weights")
                mimi = loaders.get_mimi(weights_dir / loaders.MIMI_NAME, device)
                other_mimi = loaders.get_mimi(weights_dir / loaders.MIMI_NAME, device)
                text_tokenizer = sentencepiece.SentencePieceProcessor(str(weights_dir / loaders.TEXT_TOKENIZER_NAME))
                lm = loaders.get_moshi_lm(weights_dir / loaders.MOSHI_NAME, device=device)
                
                voice_dir = _get_voice_prompt_dir(None, loaders.DEFAULT_REPO)
                self.state = ServerState(mimi=mimi, other_mimi=other_mimi, text_tokenizer=text_tokenizer, lm=lm, device=device, voice_prompt_dir=voice_dir)
                self.state.warmup()
                self.progress = "Ready"
                logger.info("🚀 BACKEND READY!")
            except Exception as e:
                self.progress = f"Error: {e}"
                logger.error(f"Load failed: {e}")

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
            return {"ready": self.state is not None, "progress": self.progress}

        @fastapi_app.websocket("/api/chat")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            if self.state is None:
                await websocket.close(code=1001)
                return

            logger.info("Accepted connection")
            query = websocket.query_params
            voice_prompt = query.get("voice_prompt")
            voice_prompt_path = None
            
            if voice_prompt:
                if voice_prompt == "pepper.pt":
                    voice_prompt_path = "/root/weights/pepper.pt"
                else:
                    voice_prompt_path = os.path.join(self.state.voice_prompt_dir, voice_prompt)
            
            if voice_prompt_path and os.path.exists(voice_prompt_path):
                self.state.lm_gen.load_voice_prompt_embeddings(voice_prompt_path) if voice_prompt_path.endswith('.pt') else self.state.lm_gen.load_voice_prompt(voice_prompt_path)
            
            self.state.lm_gen.text_prompt_tokens = self.state.text_tokenizer.encode(wrap_with_system_tags(query.get("text_prompt", ""))) if query.get("text_prompt") else None
            seed_all(int(query.get("seed", 42)))

            close = False
            async def is_alive(): return not (close or websocket.client_state == WebSocketState.DISCONNECTED)

            try:
                async with self.state.lock:
                    self.state.mimi.reset_streaming()
                    self.state.other_mimi.reset_streaming()
                    self.state.lm_gen.reset_streaming()
                    
                    logger.info("Starting system prompts...")
                    await self.state.lm_gen.step_system_prompts_async(self.state.mimi, is_alive=is_alive)
                    self.state.mimi.reset_streaming()
                    
                    await websocket.send_bytes(b"\x00") # Handshake sent
                    logger.info("Handshake sent. Phase: Conversation")

                    opus_writer = sphn.OpusStreamWriter(self.state.mimi.sample_rate)
                    opus_reader = sphn.OpusStreamReader(self.state.mimi.sample_rate)

                    async def recv_loop():
                        nonlocal close
                        try:
                            while not close:
                                msg = await websocket.receive()
                                if msg["type"] == "websocket.disconnect": break
                                if "bytes" in msg and len(msg["bytes"]) > 0:
                                    if msg["bytes"][0] == 1: opus_reader.append_bytes(msg["bytes"][1:])
                                    elif msg["bytes"][0] == 6: await websocket.send_bytes(b"\x06")
                        except: pass
                        finally: close = True

                    async def send_loop():
                        nonlocal close
                        try:
                            while not close:
                                await asyncio.sleep(0.005)
                                bytes_to_send = opus_writer.read_bytes()
                                if len(bytes_to_send) > 0:
                                    await websocket.send_bytes(b"\x01" + bytes_to_send)
                        except: pass
                        finally: close = True

                    async def opus_loop():
                        nonlocal close
                        try:
                            all_pcm_data = None
                            frame_count = 0
                            while not close:
                                await asyncio.sleep(0.005)
                                pcm = opus_reader.read_pcm()
                                if pcm is None or pcm.shape[-1] == 0: continue
                                
                                all_pcm_data = pcm if all_pcm_data is None else np.concatenate((all_pcm_data, pcm))
                                
                                while all_pcm_data is not None and all_pcm_data.shape[-1] >= self.state.frame_size:
                                    chunk_np = all_pcm_data[:self.state.frame_size]
                                    all_pcm_data = all_pcm_data[self.state.frame_size:]
                                    
                                    # RMS for volume diagnostics
                                    rms = np.sqrt(np.mean(chunk_np**2))
                                    
                                    chunk = torch.from_numpy(chunk_np).to(device=self.state.device, dtype=torch.float32)[None, None]
                                    with torch.no_grad():
                                        codes = self.state.mimi.encode(chunk)
                                        _ = self.state.other_mimi.encode(chunk)
                                        for c in range(codes.shape[-1]):
                                            tokens = self.state.lm_gen.step(codes[:, :, c: c + 1])
                                            if tokens is None: continue
                                            # Decode audio
                                            main_pcm = self.state.mimi.decode(tokens[:, 1:9])
                                            _ = self.state.other_mimi.decode(tokens[:, 1:9])
                                            
                                            output_pcm = main_pcm[0, 0].detach().cpu().numpy()
                                            out_rms = np.sqrt(np.mean(output_pcm**2))
                                            
                                            # Optional: Apply subtle gain if model output is too quiet
                                            output_pcm = output_pcm * 1.5 
                                            
                                            opus_writer.append_pcm(output_pcm)
                                            
                                            # Decode text
                                            text_token = tokens[0, 0, 0].item()
                                            if text_token not in (0, 3):
                                                text = self.state.text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                                                await websocket.send_bytes(b"\x02" + text.encode("utf8"))
                                    
                                    frame_count += 1
                                    if frame_count % 50 == 0:
                                        logger.info(f"Stepped {frame_count} frames. In RMS: {rms:.6f}, Out RMS: {out_rms:.6f}")
                        except Exception as e:
                            logger.error(f"Opus Error: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        finally: close = True

                    tasks = [asyncio.create_task(recv_loop()), asyncio.create_task(send_loop()), asyncio.create_task(opus_loop())]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending: t.cancel()
            except Exception as e:
                logger.error(f"Terminal error: {e}")
            finally:
                logger.info("Session closed")
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

@app.function(image=image, gpu="A100", volumes={"/root/weights": volume}, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def generate_pepper_embedding():
    from moshi.server import ServerState, torch_auto_device, seed_all, _get_voice_prompt_dir
    from moshi.models import loaders
    import sentencepiece
    import torch
    
    weights_dir = Path("/root/weights")
    pepper_wav = Path("/root/pepper.wav")
    pepper_pt = weights_dir / "pepper.pt"
    
    if pepper_pt.exists():
        logger.info("pepper.pt already exists, skipping generation.")
        return

    logger.info("Generating pepper.pt speaker embedding...")
    device = torch_auto_device("cuda")
    seed_all(42424242)
    
    mimi = loaders.get_mimi(weights_dir / loaders.MIMI_NAME, device)
    other_mimi = loaders.get_mimi(weights_dir / loaders.MIMI_NAME, device)
    text_tokenizer = sentencepiece.SentencePieceProcessor(str(weights_dir / loaders.TEXT_TOKENIZER_NAME))
    lm = loaders.get_moshi_lm(weights_dir / loaders.MOSHI_NAME, device=device)
    
    # We set save_voice_prompt_embeddings to True to capture the embedding
    state = ServerState(mimi=mimi, other_mimi=other_mimi, text_tokenizer=text_tokenizer, lm=lm, device=device, voice_prompt_dir=None, save_voice_prompt_embeddings=True)
    
    # Condition the model with the pepper.wav
    state.lm_gen.load_voice_prompt(str(pepper_wav))
    
    # Running warmup/system prompts will trigger the save if save_voice_prompt_embeddings is True
    # In moshi/models/lm.py: _step_voice_prompt_core saves to [splitext(self.voice_prompt)[0] + ".pt"]
    # So it will save to /root/pepper.pt
    # We should move it to /root/weights/pepper.pt
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(state.lm_gen.step_system_prompts_async(state.mimi))
    
    generated_pt = Path("/root/pepper.pt")
    if generated_pt.exists():
        import shutil
        shutil.move(str(generated_pt), str(pepper_pt))
        logger.info(f"Successfully generated and moved {pepper_pt}")
        volume.commit()
    else:
        logger.error("Failed to generate pepper.pt")
