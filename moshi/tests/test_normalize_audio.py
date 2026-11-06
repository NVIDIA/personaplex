"""Regression tests for normalize_audio channel layout.

``load_audio`` returns channels-first audio ``(C, T)`` from sphn, and
``load_voice_prompt`` passes it straight into ``normalize_audio``.
pyloudnorm expects ``(samples,)`` or ``(samples, channels)``, so a stereo
voice prompt ``(2, T)`` was misread as 2 samples with T channels and crashed
with ``ValueError: Audio must have five channels or less``.
"""

import numpy as np
import pyloudnorm as pyln

from moshi.models.lm import normalize_audio

SR = 24000
TARGET = -24.0


def _sine(freq, n=SR):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t)


def _loudness(wav):
    meter_input = wav.T if wav.ndim == 2 else wav
    return pyln.Meter(SR).integrated_loudness(meter_input)


def test_mono_1d_is_normalized():
    out = normalize_audio(_sine(440), SR, TARGET)
    assert out.ndim == 1
    assert abs(_loudness(out) - TARGET) < 0.05


def test_mono_single_channel_row_is_normalized():
    # Existing behavior: a (1, T) row is squeezed to (T,) before metering;
    # load_voice_prompt re-adds the channel dim afterwards.
    out = normalize_audio(_sine(440)[None, :], SR, TARGET)
    assert out.shape == (SR,)
    assert abs(_loudness(out) - TARGET) < 0.05


def test_stereo_channels_first_no_longer_crashes():
    # Previously raised ValueError: Audio must have five channels or less.
    stereo = np.stack([_sine(440), _sine(220)])
    out = normalize_audio(stereo, SR, TARGET)
    assert out.shape == stereo.shape
    assert abs(_loudness(out) - TARGET) < 0.05
