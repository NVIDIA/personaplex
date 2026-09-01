"""Compare sequential forced installation with batched M+W prefill.

Run on two 48-GB GPUs. Each PersonaPlex lives in a separate process so CUDA
graphs and streaming state cannot contaminate the other implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import wave

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from moshi.models import loaders
from moshi.seamless.model_slot import ProcessPersonaPlexSlot


def load_pcm(path: Path, expected_sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != expected_sample_rate
        ):
            raise ValueError("input must be mono 16-bit PCM at the model sample rate")
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def audio_frame(pcm: np.ndarray, seq: int, frame_size: int) -> np.ndarray:
    start = (seq * frame_size) % pcm.size
    indexes = (np.arange(frame_size) + start) % pcm.size
    return pcm[indexes].copy()


async def run(args: argparse.Namespace) -> dict[str, object]:
    mimi = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    moshi = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    archive = Path(hf_hub_download(args.hf_repo, "voices.tgz"))
    voice = str(archive.parent / "voices" / args.voice)
    slots = tuple(
        ProcessPersonaPlexSlot(
            name=name,
            device=f"cuda:{index}",
            mimi_weight=mimi,
            moshi_weight=moshi,
            tokenizer_path=tokenizer,
            process_rank=index,
            process_world_size=2,
        )
        for index, name in enumerate(("sequential", "batched"))
    )
    try:
        startup_started = time.perf_counter()
        await asyncio.gather(*(slot.warmup() for slot in slots))
        startup_s = time.perf_counter() - startup_started
        sequential, batched = slots
        pcm = load_pcm(args.input_audio, sequential.sample_rate)
        role = "You are a helpful customer-service agent. Continue naturally."
        memory_prompt = (
            f"{role}\n\n"
            "This is an ongoing voice conversation; do not greet or restart.\n"
            "<conversation_memory>\n"
            "User | preferred project = Aurora | status = current\n"
            "</conversation_memory>\n"
            "Newer raw duplex context overrides this memory on conflict."
        )

        # Generate one exact duplex ledger before resetting this process as the
        # sequential oracle.
        await sequential.start_session(role, voice, 1)
        trace = []
        for seq in range(args.frames):
            result = await sequential.active_step(
                seq,
                audio_frame(pcm, seq, sequential.frame_size),
            )
            trace.append(result.frame)
            if (
                result.emitted_seq is not None
                and result.emitted_seq >= 0
                and result.output_pcm is not None
            ):
                emitted = trace[result.emitted_seq]
                emitted.played = True
                emitted.played_pcm = result.output_pcm.copy()

        sequential_prefix_started = time.perf_counter()
        await sequential.start_session(memory_prompt, voice, 1)
        sequential_prefix_s = time.perf_counter() - sequential_prefix_started
        sequential_suffix_started = time.perf_counter()
        sequential_result = await sequential.replay(trace)
        sequential_suffix_s = time.perf_counter() - sequential_suffix_started

        batched_prefix_started = time.perf_counter()
        await batched.start_session(
            memory_prompt,
            voice,
            args.text_prefill_chunk_tokens,
        )
        batched_prefix_s = time.perf_counter() - batched_prefix_started
        batched_suffix_started = time.perf_counter()
        batched_results = []
        for start in range(0, len(trace), args.duplex_prefill_frames):
            batched_results.append(
                await batched.prefill_duplex(
                    trace[start : start + args.duplex_prefill_frames]
                )
            )
        batched_suffix_s = time.perf_counter() - batched_suffix_started
        assert batched_results

        await sequential.set_seed(args.next_seed)
        await batched.set_seed(args.next_seed)
        next_pcm = audio_frame(pcm, args.frames, sequential.frame_size)
        sequential_next, batched_next = await asyncio.gather(
            sequential.active_step(args.frames, next_pcm),
            batched.active_step(args.frames, next_pcm),
        )
        next_target_equal = bool(
            np.array_equal(
                sequential_next.frame.assistant_tokens,
                batched_next.frame.assistant_tokens,
            )
        )
        next_user_equal = bool(
            np.array_equal(
                sequential_next.frame.user_codes,
                batched_next.frame.user_codes,
            )
        )
        next_pcm_mse = None
        if (
            sequential_next.output_pcm is not None
            and batched_next.output_pcm is not None
        ):
            next_pcm_mse = float(
                np.mean(np.square(sequential_next.output_pcm - batched_next.output_pcm))
            )

        sequential_total_s = sequential_prefix_s + sequential_suffix_s
        batched_total_s = batched_prefix_s + batched_suffix_s
        return {
            "schema_version": 1,
            "method": "process-isolated sequential oracle vs batched M+W prefill",
            "devices": [slot.device_name for slot in slots],
            "frames": args.frames,
            "audio_duration_s": args.frames / 12.5,
            "startup_s": startup_s,
            "sequential": {
                "text_memory_prefix_s": sequential_prefix_s,
                "duplex_suffix_s": sequential_suffix_s,
                "total_install_state_s": sequential_total_s,
                "user_code_mismatches": sequential_result.user_code_mismatches,
                "target_mismatches": sequential_result.target_mismatches,
                "decoder_mismatches": sequential_result.decoder_mismatches,
            },
            "batched": {
                "text_memory_prefix_s": batched_prefix_s,
                "duplex_suffix_s": batched_suffix_s,
                "total_install_state_s": batched_total_s,
                "text_chunk_tokens": args.text_prefill_chunk_tokens,
                "duplex_chunk_frames": args.duplex_prefill_frames,
                "user_code_mismatches": sum(
                    result.user_code_mismatches for result in batched_results
                ),
                "target_mismatches": sum(
                    result.target_mismatches for result in batched_results
                ),
                "decoder_mismatches": sum(
                    result.decoder_mismatches for result in batched_results
                ),
            },
            "speedup": {
                "text_memory_prefix": sequential_prefix_s / batched_prefix_s,
                "duplex_suffix": sequential_suffix_s / batched_suffix_s,
                "total_install_state": sequential_total_s / batched_total_s,
            },
            "next_step_equivalence": {
                "user_codes_equal": next_user_equal,
                "assistant_targets_equal": next_target_equal,
                "output_pcm_mse": next_pcm_mse,
            },
        }
    finally:
        for slot in slots:
            slot.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--voice", default="NATF2.pt")
    parser.add_argument(
        "--input-audio",
        type=Path,
        default=Path("assets/test/input_service.wav"),
    )
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--text-prefill-chunk-tokens", type=int, default=64)
    parser.add_argument("--duplex-prefill-frames", type=int, default=64)
    parser.add_argument("--next-seed", type=int, default=42424242)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two visible GPUs are required")
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
