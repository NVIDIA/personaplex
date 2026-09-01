from __future__ import annotations

import asyncio
import hashlib
import unittest

import numpy as np

from moshi.seamless.controller import SeamlessInstallController
from moshi.seamless.memory import NullMemoryWorker
from moshi.seamless.types import (
    CompactResult,
    ControllerConfig,
    DuplexFrame,
    ReplayResult,
    StepResult,
)


class FakeSlot:
    def __init__(
        self,
        name: str,
        *,
        replay_delay: float = 0.0,
        codec_mismatch_seqs: set[int] | None = None,
    ) -> None:
        self.name = name
        self.replay_delay = replay_delay
        self.codec_mismatch_seqs = codec_mismatch_seqs or set()
        self.prompts: list[str] = []
        self.active_seqs: list[int] = []
        self.replayed_seqs: list[int] = []
        self.prefilled_seqs: list[int] = []

    async def start_session(
        self,
        prompt: str,
        voice_prompt: str,
        text_prefill_chunk_size: int = 64,
    ) -> None:
        del text_prefill_chunk_size
        self.prompts.append(f"{voice_prompt}|{prompt}")

    async def active_step(self, seq: int, pcm: np.ndarray) -> StepResult:
        self.active_seqs.append(seq)
        user = np.full(8, seq % 16, dtype=np.int64)
        assistant = np.arange(9, dtype=np.int64) + seq
        frame = DuplexFrame(
            seq=seq,
            pcm=pcm,
            user_codes=user,
            assistant_tokens=assistant,
            target_digest=hashlib.sha256(assistant.tobytes()).hexdigest(),
        )
        return StepResult(
            frame=frame,
            output_pcm=np.zeros_like(pcm),
            output_text="",
            emitted_seq=seq,
            context_frames=seq + 1,
        )

    async def replay(self, frames: list[DuplexFrame]) -> ReplayResult:
        if self.replay_delay:
            await asyncio.sleep(self.replay_delay)
        self.replayed_seqs.extend(frame.seq for frame in frames)
        mismatches = [
            frame.seq for frame in frames if frame.seq in self.codec_mismatch_seqs
        ]
        return ReplayResult(
            start_seq=frames[0].seq,
            end_seq=frames[-1].seq,
            frames=len(frames),
            user_code_mismatches=len(mismatches),
            target_mismatches=0,
            last_user_mismatch_seq=mismatches[-1] if mismatches else None,
            decoder_frames_compared=len(frames),
            decoder_mismatches=0,
            last_decoder_compared_seq=frames[-1].seq,
        )

    async def prefill_duplex(self, frames: list[DuplexFrame]) -> ReplayResult:
        if self.replay_delay:
            await asyncio.sleep(self.replay_delay)
        self.prefilled_seqs.extend(frame.seq for frame in frames)
        mismatches = [
            frame.seq for frame in frames if frame.seq in self.codec_mismatch_seqs
        ]
        return ReplayResult(
            start_seq=frames[0].seq,
            end_seq=frames[-1].seq,
            frames=len(frames),
            user_code_mismatches=len(mismatches),
            target_mismatches=0,
            last_user_mismatch_seq=mismatches[-1] if mismatches else None,
            decoder_frames_compared=len(frames),
            decoder_mismatches=0,
            last_decoder_compared_seq=frames[-1].seq,
        )


class BlockingMemoryWorker:
    auto_compaction = True

    def __init__(self, *, cutoff_delta: int = 0) -> None:
        self.release = asyncio.Event()
        self.compact_started = asyncio.Event()
        self.cutoff_delta = cutoff_delta
        self.audio_seqs: list[int] = []

    async def start_session(self, session_id: str, role_prompt: str) -> None:
        del session_id, role_prompt

    async def push_audio(self, session_id: str, seq: int, pcm: np.ndarray) -> None:
        del session_id, pcm
        self.audio_seqs.append(seq)

    async def push_assistant_text(
        self, session_id: str, seq: int, text_piece: str
    ) -> None:
        del session_id, seq, text_piece

    async def compact(self, request) -> CompactResult:
        self.compact_started.set()
        await self.release.wait()
        return CompactResult(
            request_id=request.request_id,
            parent_version=request.parent_version,
            cutoff_seq=request.cutoff_hint_seq + self.cutoff_delta,
            memory="User's preferred project is Aurora.",
        )

    async def close_session(self, session_id: str) -> None:
        del session_id


def test_config() -> ControllerConfig:
    return ControllerConfig(
        context_capacity_frames=8,
        trigger_reserve_frames=5,
        prefill_batch_frames=2,
        text_prefill_chunk_tokens=2,
        switch_max_lag_frames=0,
        safe_silence_frames=1,
        force_switch_after_s=0.05,
        install_timeout_s=2.0,
        ledger_max_frames=32,
        codec_convergence_frames=0,
        decoder_convergence_frames=0,
    )


class SeamlessControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_inference_continues_during_compaction_then_swaps(
        self,
    ) -> None:
        active = FakeSlot("gpu-2")
        shadow = FakeSlot("gpu-3", replay_delay=0.001)
        memory = BlockingMemoryWorker()
        controller = SeamlessInstallController(
            (active, shadow), memory, config=test_config()
        )
        await controller.start_session(
            session_id="s1", role_prompt="Be helpful.", voice_prompt="voice.pt"
        )
        self.assertEqual(len(active.prompts), 1)
        self.assertEqual(len(shadow.prompts), 1)

        for _ in range(3):
            await controller.process_pcm(np.zeros(4, dtype=np.float32))
        await asyncio.wait_for(memory.compact_started.wait(), timeout=0.5)
        self.assertEqual(controller.status()["phase"], "compacting")

        # GPU-2 remains the response path while GPU-1 compaction is blocked.
        await controller.process_pcm(np.zeros(4, dtype=np.float32))
        self.assertEqual(active.active_seqs, [0, 1, 2, 3])
        self.assertEqual(controller.active_slot.name, "gpu-2")

        memory.release.set()
        swap = await controller.wait_for_install()
        self.assertIsNotNone(swap)
        self.assertEqual(controller.active_slot.name, "gpu-3")
        self.assertIn(3, shadow.prefilled_seqs)
        self.assertEqual(shadow.replayed_seqs, [])
        self.assertIn("conversation_memory", shadow.prompts[-1])
        self.assertGreaterEqual(swap.timings["critical_section_s"], 0)
        self.assertGreaterEqual(swap.timings["trigger_to_swap_s"], 0)
        await controller.close_session()

    async def test_manual_installs_ping_pong_between_resident_slots(self) -> None:
        slot_a = FakeSlot("slot-a")
        slot_b = FakeSlot("slot-b")
        controller = SeamlessInstallController(
            (slot_a, slot_b), NullMemoryWorker(), config=test_config()
        )
        await controller.start_session(
            session_id="s2", role_prompt="Role", voice_prompt="voice.pt"
        )
        for _ in range(3):
            await controller.process_pcm(np.zeros(4, dtype=np.float32))
        self.assertTrue(await controller.request_install(memory="Memory v1"))
        await controller.wait_for_install()
        self.assertEqual(controller.active_slot.name, "slot-b")

        for _ in range(2):
            await controller.process_pcm(np.zeros(4, dtype=np.float32))
        self.assertTrue(await controller.request_install(memory="Memory v2"))
        swap = await controller.wait_for_install()
        self.assertEqual(controller.active_slot.name, "slot-a")
        self.assertEqual(swap.version, 2)
        self.assertIn("Memory v2", slot_a.prompts[-1])
        await controller.close_session()

    async def test_wrong_compactor_cutoff_never_replaces_active_slot(self) -> None:
        slot_a = FakeSlot("slot-a")
        slot_b = FakeSlot("slot-b")
        memory = BlockingMemoryWorker(cutoff_delta=-1)
        controller = SeamlessInstallController(
            (slot_a, slot_b), memory, config=test_config()
        )
        await controller.start_session(
            session_id="s3", role_prompt="Role", voice_prompt="voice.pt"
        )
        for _ in range(3):
            await controller.process_pcm(np.zeros(4, dtype=np.float32))
        await memory.compact_started.wait()
        memory.release.set()
        await controller.wait_for_install()

        self.assertEqual(controller.active_slot.name, "slot-a")
        self.assertEqual(controller.status()["phase"], "active")
        self.assertIn("cutoff", controller.status()["last_error"])
        await controller.close_session()

    async def test_codec_mismatch_must_converge_before_swap(self) -> None:
        config = test_config()
        config.codec_convergence_frames = 3
        slot_a = FakeSlot("slot-a")
        slot_b = FakeSlot("slot-b", codec_mismatch_seqs={1})
        controller = SeamlessInstallController(
            (slot_a, slot_b), NullMemoryWorker(), config=config
        )
        await controller.start_session(
            session_id="s4", role_prompt="Role", voice_prompt="voice.pt"
        )
        for _ in range(5):
            await controller.process_pcm(np.zeros(4, dtype=np.float32))
        await controller.request_install(memory="Memory", cutoff_seq=0)
        swap = await controller.wait_for_install()
        self.assertIsNotNone(swap)
        self.assertEqual(controller.active_slot.name, "slot-b")
        await controller.close_session()

    async def test_memory_worker_failure_does_not_stop_active_audio(self) -> None:
        class FailingMemory(BlockingMemoryWorker):
            async def push_audio(
                self, session_id: str, seq: int, pcm: np.ndarray
            ) -> None:
                del session_id, seq, pcm
                raise RuntimeError("GPU-1 unavailable")

        slot_a = FakeSlot("slot-a")
        slot_b = FakeSlot("slot-b")
        controller = SeamlessInstallController(
            (slot_a, slot_b), FailingMemory(), config=test_config()
        )
        await controller.start_session(
            session_id="s5", role_prompt="Role", voice_prompt="voice.pt"
        )
        for _ in range(4):
            await controller.process_pcm(np.zeros(4, dtype=np.float32))
        self.assertEqual(slot_a.active_seqs, [0, 1, 2, 3])
        self.assertFalse(controller.status()["memory_worker_healthy"])
        self.assertEqual(controller.active_slot.name, "slot-a")
        await controller.close_session()


if __name__ == "__main__":
    unittest.main()
