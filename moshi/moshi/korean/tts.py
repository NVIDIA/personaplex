# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from typing import Generator, Optional

import numpy as np
import torch

from .voices import KOREAN_VOICES, DEFAULT_KOREAN_VOICE, get_voice

logger = logging.getLogger(__name__)

# PersonaPlex native sample rate
TARGET_SAMPLE_RATE = 24000


class KoreanTTS:
    """Korean TTS using CosyVoice2-0.5B for text-to-speech synthesis."""

    def __init__(
        self,
        model_id: str = "FunAudioLLM/CosyVoice2-0.5B",
        device: str = "cuda",
    ):
        self.device = device
        self.model_id = model_id
        self._model = None
        self._model_sample_rate: Optional[int] = None
        logger.info(f"KoreanTTS will load model={model_id} on device={device}")

    def _ensure_model(self):
        """Lazy-load the CosyVoice2 model on first use."""
        if self._model is not None:
            return

        from cosyvoice import CosyVoice2

        logger.info(f"Loading CosyVoice2 model: {self.model_id}")
        self._model = CosyVoice2(self.model_id, load_jit=True, load_trt=False)
        self._model_sample_rate = self._model.sample_rate
        logger.info(
            f"CosyVoice2 loaded, native sample_rate={self._model_sample_rate}"
        )

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_KOREAN_VOICE,
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize Korean text to audio, yielding chunks as they're generated.

        Args:
            text: Korean text to synthesize.
            voice: Voice key from KOREAN_VOICES.

        Yields:
            Audio chunks as float32 numpy arrays at TARGET_SAMPLE_RATE (24kHz).
        """
        self._ensure_model()

        voice_config = get_voice(voice)
        speaker_id = voice_config.speaker_id

        logger.debug(f"Synthesizing: '{text[:50]}...' with voice={voice}")

        for chunk_result in self._model.inference_zero_shot_streaming(
            tts_text=text,
            prompt_text="",
            prompt_speech_16k=None,
            stream=True,
            speed=1.0,
            speaker_id=speaker_id,
        ):
            if isinstance(chunk_result, dict) and "tts_speech" in chunk_result:
                audio_tensor = chunk_result["tts_speech"]
            elif isinstance(chunk_result, torch.Tensor):
                audio_tensor = chunk_result
            else:
                continue

            audio_np = audio_tensor.cpu().numpy().squeeze()
            if audio_np.ndim == 0 or len(audio_np) == 0:
                continue

            audio_np = self._resample(
                audio_np, self._model_sample_rate, TARGET_SAMPLE_RATE
            )
            yield audio_np

    def synthesize_full(
        self,
        text: str,
        voice: str = DEFAULT_KOREAN_VOICE,
    ) -> np.ndarray:
        """Synthesize Korean text to a single audio array.

        Args:
            text: Korean text to synthesize.
            voice: Voice key from KOREAN_VOICES.

        Returns:
            Complete audio as float32 numpy array at TARGET_SAMPLE_RATE (24kHz).
        """
        chunks = list(self.synthesize(text, voice))
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)
