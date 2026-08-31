"""Exact duplex frame ledger shared by active and shadow PersonaPlex slots."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np

from .types import DuplexFrame


class DuplexFrameLedger:
    """Monotonic replay log with a playout watermark and install pin."""

    def __init__(self, max_frames: int = 20_000) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.max_frames = max_frames
        self._frames: OrderedDict[int, DuplexFrame] = OrderedDict()
        self._next_seq = 0
        self._playout_watermark = -1
        self._pin_start: int | None = None

    @property
    def next_seq(self) -> int:
        return self._next_seq

    @property
    def first_seq(self) -> int:
        return next(iter(self._frames), self._next_seq)

    @property
    def latest_seq(self) -> int:
        return self._next_seq - 1

    @property
    def playout_watermark(self) -> int:
        return self._playout_watermark

    def __len__(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()
        self._next_seq = 0
        self._playout_watermark = -1
        self._pin_start = None

    def append(self, frame: DuplexFrame) -> None:
        if frame.seq != self._next_seq:
            raise ValueError(f"expected frame {self._next_seq}, got {frame.seq}")
        self._frames[frame.seq] = frame
        self._next_seq += 1
        self._trim()

    def mark_played(self, seq: int, pcm: np.ndarray) -> None:
        if seq not in self._frames:
            raise KeyError(f"cannot mark missing frame {seq} as played")
        if seq > self._playout_watermark + 1:
            raise ValueError(
                f"playout gap: watermark={self._playout_watermark}, seq={seq}"
            )
        frame = self._frames[seq]
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        frame.played = True
        frame.played_rms = (
            float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        )
        frame.played_pcm = samples.copy()
        self._playout_watermark = max(self._playout_watermark, seq)

    def pin(self, start_seq: int) -> None:
        if start_seq < self.first_seq or start_seq > self._next_seq:
            raise ValueError(f"cannot pin unavailable frame {start_seq}")
        self._pin_start = start_seq

    def unpin(self) -> None:
        self._pin_start = None
        self._trim()

    def slice(self, start_seq: int, end_seq: int | None = None) -> list[DuplexFrame]:
        end = self._next_seq if end_seq is None else end_seq
        if not self.first_seq <= start_seq <= end <= self._next_seq:
            raise ValueError(
                f"unavailable ledger slice {start_seq}:{end}; "
                f"available={self.first_seq}:{self._next_seq}"
            )
        return [self._frames[index] for index in range(start_seq, end)]

    def digest(self, start_seq: int, end_seq: int | None = None) -> str:
        digest = hashlib.sha256()
        for frame in self.slice(start_seq, end_seq):
            digest.update(frame.seq.to_bytes(8, "big", signed=False))
            digest.update(frame.user_codes.tobytes())
            digest.update(frame.assistant_tokens.tobytes())
        return digest.hexdigest()

    def is_safe_boundary(
        self,
        silence_frames: int,
        rms_threshold: float,
    ) -> bool:
        if silence_frames == 0:
            return True
        end = self._playout_watermark + 1
        start = end - silence_frames
        if start < self.first_seq or end <= 0:
            return False
        frames = self.slice(start, end)
        return all(
            frame.played
            and frame.played_rms is not None
            and frame.played_rms <= rms_threshold
            for frame in frames
        )

    def _trim(self) -> None:
        while len(self._frames) > self.max_frames:
            first = self.first_seq
            if self._pin_start is not None and first >= self._pin_start:
                # Installation must not interrupt the live audio path merely
                # because prefill/replay is slow. Temporarily exceed the soft
                # bound, then trim immediately when the install releases its
                # pin. The controller's install timeout bounds this growth.
                return
            self._frames.popitem(last=False)


def digest_frames(frames: Iterable[DuplexFrame]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.target_digest.encode("ascii"))
    return digest.hexdigest()
