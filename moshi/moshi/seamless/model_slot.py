"""Resident PersonaPlex slot used as either active or shadow backend."""

from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import sentencepiece
import torch

from ..models import LMGen, loaders
from .types import DuplexFrame, ReplayResult, StepResult


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


class PersonaPlexSlotEngine:
    """Synchronous GPU engine; call it only through ``AsyncPersonaPlexSlot``."""

    def __init__(
        self,
        *,
        name: str,
        device: str | torch.device,
        mimi_weight: str,
        moshi_weight: str,
        tokenizer_path: str,
        cpu_offload: bool = False,
        decoder_mse_threshold: float = 1e-7,
    ) -> None:
        self.name = name
        self.device = torch.device(device)
        self.mimi = loaders.get_mimi(mimi_weight, self.device)
        self.other_mimi = loaders.get_mimi(mimi_weight, self.device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        self.lm = loaders.get_moshi_lm(
            moshi_weight,
            device=self.device,
            cpu_offload=cpu_offload,
        ).eval()
        self.lm_gen = LMGen(
            self.lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=self.device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
        )
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.decoder_mse_threshold = decoder_mse_threshold
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self._voice_prompt: str | None = None
        self._started = False
        self._replay_frames: dict[int, DuplexFrame] = {}

    @property
    def max_delay(self) -> int:
        return int(self.lm_gen.max_delay)

    @torch.no_grad()
    def warmup(self) -> None:
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for index in range(codes.shape[-1]):
                output = self.lm_gen.step(codes[:, :, index : index + 1])
                if output is not None:
                    _ = self.mimi.decode(output[:, 1:9])
                    _ = self.other_mimi.decode(output[:, 1:9])
        self._sync()
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    @torch.no_grad()
    def start_session(self, prompt: str, voice_prompt: str) -> None:
        if self._voice_prompt != voice_prompt:
            if voice_prompt.endswith(".pt"):
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt)
            self._voice_prompt = voice_prompt
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.text_prompt_tokens = self.tokenizer.encode(
            wrap_with_system_tags(prompt)
        )
        self.lm_gen.step_system_prompts(self.mimi)
        # Match the official web server: conversation input starts with a clean
        # Mimi encoder while the LM retains its system/voice prefix state.
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self._sync()
        self._started = True
        self._replay_frames.clear()

    @torch.no_grad()
    def active_step(self, seq: int, pcm: np.ndarray) -> StepResult:
        self._require_started()
        pcm_array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm_array.shape[0] != self.frame_size:
            raise ValueError(
                f"{self.name}: expected {self.frame_size} PCM samples, "
                f"got {pcm_array.shape[0]}"
            )
        chunk = torch.from_numpy(pcm_array.copy()).to(self.device)[None, None]
        user = self.mimi.encode(chunk)
        _ = self.other_mimi.encode(chunk)
        if user.shape[-1] != 1:
            raise RuntimeError(
                f"{self.name}: one PCM frame encoded to {user.shape[-1]} code frames"
            )
        output = self.lm_gen.step(user)
        target = self._last_target_tokens()
        digest = hashlib.sha256(target.tobytes()).hexdigest()
        frame = DuplexFrame(
            seq=seq,
            pcm=pcm_array,
            user_codes=user[0, :, 0].detach().to("cpu", dtype=torch.long).numpy(),
            assistant_tokens=target,
            target_digest=digest,
        )
        output_pcm: np.ndarray | None = None
        output_text = ""
        emitted_seq: int | None = None
        if output is not None:
            decoded = self.mimi.decode(output[:, 1:9])
            _ = self.other_mimi.decode(output[:, 1:9])
            output_pcm = decoded[0, 0].detach().to("cpu", dtype=torch.float32).numpy()
            token = int(output[0, 0, 0])
            if token not in (0, 3):
                output_text = self.tokenizer.id_to_piece(token).replace("▁", " ")
            emitted_seq = seq - self.max_delay
        return StepResult(
            frame=frame,
            output_pcm=output_pcm,
            output_text=output_text,
            emitted_seq=emitted_seq,
            context_frames=self.context_frames(),
        )

    @torch.no_grad()
    def replay(self, frames: list[DuplexFrame]) -> ReplayResult:
        self._require_started()
        if not frames:
            raise ValueError("cannot replay an empty batch")
        user_mismatches = 0
        target_mismatches = 0
        last_user_mismatch_seq: int | None = None
        decoder_frames_compared = 0
        decoder_mismatches = 0
        last_decoder_mismatch_seq: int | None = None
        last_decoder_compared_seq: int | None = None
        for frame in frames:
            self._replay_frames[frame.seq] = frame
        for frame in frames:
            chunk = torch.from_numpy(frame.pcm.copy()).to(self.device)[None, None]
            user = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            observed_user = user[0, :, 0].detach().to("cpu", dtype=torch.long).numpy()
            if not np.array_equal(observed_user, frame.user_codes):
                user_mismatches += 1
                last_user_mismatch_seq = frame.seq

            # A public ``moshi_tokens=...`` call applies per-codebook delays,
            # so it expects an aligned assistant stream, not the target cache
            # position captured from the active model. Install must also catch
            # up frames whose delayed output has not played yet. We therefore
            # force the exact target position directly, then advance through
            # the normal LMGen step. The stored user codes are fed to the LM;
            # re-encoding PCM above solely warms the shadow Mimi state and
            # measures when its causal encoder has converged.
            forced_user = (
                torch.from_numpy(frame.user_codes.copy())
                .to(self.device, dtype=torch.long)
                .view(1, 8, 1)
            )
            self._force_next_target(frame.assistant_tokens)
            output = self.lm_gen.step(input_tokens=forced_user)
            target_mismatches += int(
                not np.array_equal(self._last_target_tokens(), frame.assistant_tokens)
            )
            if output is not None:
                decoded = self.mimi.decode(output[:, 1:9])
                _ = self.other_mimi.decode(output[:, 1:9])
                emitted_seq = frame.seq - self.max_delay
                expected_frame = self._replay_frames.pop(emitted_seq, None)
                if expected_frame is not None and expected_frame.played_pcm is not None:
                    expected_pcm = expected_frame.played_pcm
                    observed_pcm = (
                        decoded[0, 0].detach().to("cpu", dtype=torch.float32).numpy()
                    )
                    if observed_pcm.shape != expected_pcm.shape:
                        raise RuntimeError(
                            f"{self.name}: replay PCM shape mismatch at {emitted_seq}: "
                            f"{observed_pcm.shape} != {expected_pcm.shape}"
                        )
                    mse = float(np.mean(np.square(observed_pcm - expected_pcm)))
                    decoder_frames_compared += 1
                    last_decoder_compared_seq = emitted_seq
                    if mse > self.decoder_mse_threshold:
                        decoder_mismatches += 1
                        last_decoder_mismatch_seq = emitted_seq
        self._sync()
        return ReplayResult(
            start_seq=frames[0].seq,
            end_seq=frames[-1].seq,
            frames=len(frames),
            user_code_mismatches=user_mismatches,
            target_mismatches=target_mismatches,
            last_user_mismatch_seq=last_user_mismatch_seq,
            decoder_frames_compared=decoder_frames_compared,
            decoder_mismatches=decoder_mismatches,
            last_decoder_mismatch_seq=last_decoder_mismatch_seq,
            last_decoder_compared_seq=last_decoder_compared_seq,
        )

    def context_frames(self) -> int:
        state = self.lm_gen._streaming_state
        if state is None:
            return 0
        return int(state.offset)

    def _last_target_tokens(self) -> np.ndarray:
        state = self.lm_gen._streaming_state
        if state is None or state.offset <= 0:
            raise RuntimeError(f"{self.name}: LMGen has no generated target")
        position = (int(state.offset) - 1) % int(state.cache.shape[2])
        return (
            state.cache[0, :9, position]
            .detach()
            .to("cpu", dtype=torch.long)
            .numpy()
            .copy()
        )

    def _force_next_target(self, tokens: np.ndarray) -> None:
        state = self.lm_gen._streaming_state
        if state is None:
            raise RuntimeError(f"{self.name}: LMGen is not streaming")
        forced = torch.from_numpy(np.asarray(tokens, dtype=np.int64).copy()).to(
            self.device, dtype=torch.long
        )
        position = int(state.offset) % int(state.cache.shape[2])
        state.cache[:, :9, position] = forced.view(1, 9)
        state.provided[:, :9, position] = True

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(f"{self.name}: session has not been started")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class AsyncPersonaPlexSlot:
    """Serialize one resident GPU slot on its own worker thread."""

    def __init__(self, engine: PersonaPlexSlotEngine) -> None:
        self.engine = engine
        self.name = engine.name
        self.max_delay = engine.max_delay
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"personaplex-{engine.name}"
        )

    async def warmup(self) -> None:
        await self._call(self.engine.warmup)

    async def start_session(self, prompt: str, voice_prompt: str) -> None:
        await self._call(self.engine.start_session, prompt, voice_prompt)

    async def active_step(self, seq: int, pcm: np.ndarray) -> StepResult:
        return await self._call(self.engine.active_step, seq, pcm)

    async def replay(self, frames: list[DuplexFrame]) -> ReplayResult:
        return await self._call(self.engine.replay, frames)

    async def context_frames(self) -> int:
        return await self._call(self.engine.context_frames)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _call(self, function: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, function, *args)
