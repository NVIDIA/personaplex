from __future__ import annotations

import hashlib
import unittest

import numpy as np

from moshi.seamless.ledger import DuplexFrameLedger
from moshi.seamless.types import DuplexFrame


def make_frame(seq: int) -> DuplexFrame:
    assistant = np.arange(9, dtype=np.int64) + seq
    return DuplexFrame(
        seq=seq,
        pcm=np.zeros(4, dtype=np.float32),
        user_codes=np.arange(8, dtype=np.int64) + seq,
        assistant_tokens=assistant,
        target_digest=hashlib.sha256(assistant.tobytes()).hexdigest(),
    )


class DuplexFrameLedgerTests(unittest.TestCase):
    def test_pin_preserves_the_install_tail(self) -> None:
        ledger = DuplexFrameLedger(max_frames=3)
        for seq in range(3):
            ledger.append(make_frame(seq))
        ledger.pin(2)
        ledger.append(make_frame(3))
        ledger.append(make_frame(4))
        ledger.append(make_frame(5))
        self.assertEqual([frame.seq for frame in ledger.slice(2)], [2, 3, 4, 5])
        self.assertEqual(len(ledger), 4)
        ledger.unpin()
        self.assertEqual([frame.seq for frame in ledger.slice(3)], [3, 4, 5])

    def test_silence_boundary_uses_contiguous_playout(self) -> None:
        ledger = DuplexFrameLedger(max_frames=8)
        for seq in range(3):
            ledger.append(make_frame(seq))
            ledger.mark_played(seq, np.zeros(4, dtype=np.float32))
        self.assertTrue(ledger.is_safe_boundary(3, 0.001))
        ledger.append(make_frame(3))
        ledger.mark_played(3, np.ones(4, dtype=np.float32))
        self.assertFalse(ledger.is_safe_boundary(2, 0.001))


if __name__ == "__main__":
    unittest.main()
