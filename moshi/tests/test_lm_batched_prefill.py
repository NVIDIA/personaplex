from __future__ import annotations

import copy
import unittest

import torch

from moshi.models.lm import LMGen, LMModel


DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]


def tiny_lm() -> LMModel:
    return LMModel(
        delays=DELAYS,
        n_q=16,
        dep_q=8,
        card=32,
        text_card=64,
        dim=32,
        num_heads=4,
        num_layers=2,
        hidden_scale=2,
        causal=True,
        context=64,
        positional_embedding="rope",
        gating="silu",
        norm="rms_norm",
        depformer_dim=16,
        depformer_dim_feedforward=32,
        depformer_num_heads=4,
        depformer_num_layers=2,
        depformer_causal=True,
        depformer_context=8,
        depformer_gating="silu",
        depformer_pos_emb="none",
        depformer_weights_per_step=True,
    ).eval()


def lm_gen(model: LMModel) -> LMGen:
    generator = LMGen(
        model,
        device="cpu",
        use_sampling=False,
        sample_rate=80,
        frame_rate=10,
    )
    generator.streaming_forever(1)
    return generator


def force_target(generator: LMGen, target: torch.Tensor) -> None:
    state = generator._streaming_state
    assert state is not None
    position = state.offset % state.cache.shape[2]
    state.cache[:, :9, position] = target
    state.provided[:, :9, position] = True


class BatchedPrefillTests(unittest.TestCase):
    def test_batched_prefill_matches_fully_forced_steps(self) -> None:
        torch.manual_seed(7)
        sequential_model = tiny_lm()
        batched_model = copy.deepcopy(sequential_model)
        sequential = lm_gen(sequential_model)
        batched = lm_gen(batched_model)

        prompt_steps = 12
        prompt_user = torch.randint(0, 32, (1, 8, prompt_steps))
        prompt_assistant = torch.randint(0, 32, (1, 8, prompt_steps))
        prompt_text = torch.randint(0, 64, (1, prompt_steps))
        for index in range(prompt_steps):
            sequential.step(
                input_tokens=prompt_user[:, :, index : index + 1],
                moshi_tokens=prompt_assistant[:, :, index : index + 1],
                text_token=prompt_text[:, index],
            )
        batched.prefill_aligned(
            input_tokens=prompt_user,
            moshi_tokens=prompt_assistant,
            text_tokens=prompt_text,
        )

        # Cross the 64-position RingKVCache boundary. The batched path should
        # use one exact chunk up to capacity and length-one updates afterwards.
        tail_steps = 70
        tail_user = torch.randint(0, 32, (1, 8, tail_steps))
        tail_target = torch.cat(
            [
                torch.randint(0, 64, (1, 1, tail_steps)),
                torch.randint(0, 32, (1, 8, tail_steps)),
            ],
            dim=1,
        )
        sequential_emitted: list[torch.Tensor] = []
        for index in range(tail_steps):
            force_target(sequential, tail_target[:, :, index])
            output = sequential.step(input_tokens=tail_user[:, :, index : index + 1])
            self.assertIsNotNone(output)
            sequential_emitted.append(output)
        batched_emitted = batched.prefill_forced_targets(
            input_tokens=tail_user,
            target_tokens=tail_target,
        )

        self.assertIsNotNone(batched_emitted)
        torch.testing.assert_close(
            batched_emitted,
            torch.cat(sequential_emitted, dim=2),
            rtol=0,
            atol=0,
        )
        sequential_state = sequential._streaming_state
        batched_state = batched._streaming_state
        assert sequential_state is not None and batched_state is not None
        self.assertEqual(sequential_state.offset, batched_state.offset)
        torch.testing.assert_close(
            sequential_state.cache,
            batched_state.cache,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sequential_state.provided,
            batched_state.provided,
            rtol=0,
            atol=0,
        )

        for sequential_layer, batched_layer in zip(
            sequential_model.transformer.layers,
            batched_model.transformer.layers,
        ):
            sequential_attention = sequential_layer.self_attn._streaming_state
            batched_attention = batched_layer.self_attn._streaming_state
            assert sequential_attention is not None and batched_attention is not None
            self.assertEqual(
                sequential_attention.offset_cpu,
                batched_attention.offset_cpu,
            )
            torch.testing.assert_close(
                sequential_attention.kv_cache.cache,
                batched_attention.kv_cache.cache,
                # Batched SDPA changes floating-point reduction order. The
                # end-to-end token assertion below is the behavioral oracle.
                rtol=2e-2,
                atol=1e-2,
            )
            torch.testing.assert_close(
                sequential_attention.kv_cache.end_offset,
                batched_attention.kv_cache.end_offset,
                rtol=0,
                atol=0,
            )

        next_user = torch.randint(0, 32, (1, 8, 1))
        sequential_next = sequential.step(input_tokens=next_user)
        batched_next = batched.step(input_tokens=next_user)
        torch.testing.assert_close(sequential_next, batched_next, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
