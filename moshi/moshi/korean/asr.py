# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Silence detection threshold (RMS energy)
SILENCE_THRESHOLD = 0.01
# Minimum audio length in seconds before attempting transcription
MIN_AUDIO_LENGTH_SEC = 0.5
# Maximum buffer length in seconds before forcing transcription
MAX_BUFFER_LENGTH_SEC = 10.0


class KoreanASR:
    """Korean ASR using faster-whisper for speech-to-text."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        sample_rate: int = 24000,
    ):
        from faster_whisper import WhisperModel

        self.sample_rate = sample_rate
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        self._buffer: Optional[np.ndarray] = None
        self._silence_frames = 0
        # Number of consecutive silent frames to trigger transcription
        self._silence_trigger = int(0.5 * sample_rate / 480)  # ~0.5s of silence
        logger.info(f"KoreanASR initialized with model={model_size}, device={device}")

    def reset(self):
        self._buffer = None
        self._silence_frames = 0

    def _is_silent(self, audio: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < SILENCE_THRESHOLD

    def add_audio(self, audio_pcm: np.ndarray) -> Optional[str]:
        """Add audio frames and return transcription when ready.

        Accumulates audio frames and triggers transcription when silence
        is detected or the buffer is full.

        Args:
            audio_pcm: Audio samples as float32 numpy array, mono, at self.sample_rate.

        Returns:
            Transcribed Korean text, or None if more audio is needed.
        """
        if self._buffer is None:
            self._buffer = audio_pcm
        else:
            self._buffer = np.concatenate([self._buffer, audio_pcm])

        buffer_duration = len(self._buffer) / self.sample_rate

        if self._is_silent(audio_pcm):
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        should_transcribe = False
        if buffer_duration >= MAX_BUFFER_LENGTH_SEC:
            should_transcribe = True
        elif (
            self._silence_frames >= self._silence_trigger
            and buffer_duration >= MIN_AUDIO_LENGTH_SEC
        ):
            should_transcribe = True

        if should_transcribe:
            text = self.transcribe(self._buffer)
            self._buffer = None
            self._silence_frames = 0
            return text

        return None

    def transcribe(self, audio_pcm: np.ndarray) -> str:
        """Transcribe audio to Korean text.

        Args:
            audio_pcm: Audio samples as float32 numpy array, mono, at self.sample_rate.

        Returns:
            Transcribed Korean text.
        """
        segments, info = self.model.transcribe(
            audio_pcm,
            language="ko",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        result = " ".join(text_parts).strip()
        if result:
            logger.debug(f"Transcribed: {result}")
        return result
