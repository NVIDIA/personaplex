# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class KoreanVoice:
    name: str
    description: str
    gender: str
    speaker_id: str


KOREAN_VOICES: Dict[str, KoreanVoice] = {
    "ko_female_1": KoreanVoice(
        name="한국어 여성 1",
        description="Korean Female Natural",
        gender="F",
        speaker_id="korean_female_natural_1",
    ),
    "ko_female_2": KoreanVoice(
        name="한국어 여성 2",
        description="Korean Female Expressive",
        gender="F",
        speaker_id="korean_female_expressive_1",
    ),
    "ko_male_1": KoreanVoice(
        name="한국어 남성 1",
        description="Korean Male Natural",
        gender="M",
        speaker_id="korean_male_natural_1",
    ),
    "ko_male_2": KoreanVoice(
        name="한국어 남성 2",
        description="Korean Male Expressive",
        gender="M",
        speaker_id="korean_male_expressive_1",
    ),
}

DEFAULT_KOREAN_VOICE = "ko_female_1"


def get_voice(voice_key: str) -> KoreanVoice:
    if voice_key not in KOREAN_VOICES:
        raise ValueError(
            f"Unknown Korean voice '{voice_key}'. "
            f"Available voices: {list(KOREAN_VOICES.keys())}"
        )
    return KOREAN_VOICES[voice_key]


def list_voices() -> List[Dict[str, str]]:
    return [
        {
            "key": key,
            "name": voice.name,
            "description": voice.description,
            "gender": voice.gender,
        }
        for key, voice in KOREAN_VOICES.items()
    ]
