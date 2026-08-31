"""Two-GPU smoke test for exact PersonaPlex tail replay.

This is intentionally outside unittest discovery. It loads two full checkpoints
and should be run on a node with at least two 48-GB GPUs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from moshi.models import loaders
from moshi.seamless.model_slot import PersonaPlexSlotEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--voice", default="NATF2.pt")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument("--decoder-mse-threshold", type=float, default=1e-7)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("gpu_seamless_smoke requires two visible GPUs")
    if args.frames <= args.cutoff + 24:
        raise ValueError("tail must leave enough frames for codec convergence")

    mimi = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    moshi = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    archive = Path(hf_hub_download(args.hf_repo, "voices.tgz"))
    voice = archive.parent / "voices" / args.voice
    if not voice.is_file():
        raise FileNotFoundError(voice)

    active = PersonaPlexSlotEngine(
        name="active",
        device="cuda:0",
        mimi_weight=mimi,
        moshi_weight=moshi,
        tokenizer_path=tokenizer,
        decoder_mse_threshold=args.decoder_mse_threshold,
    )
    shadow = PersonaPlexSlotEngine(
        name="shadow",
        device="cuda:1",
        mimi_weight=mimi,
        moshi_weight=moshi,
        tokenizer_path=tokenizer,
        decoder_mse_threshold=args.decoder_mse_threshold,
    )
    active.warmup()
    shadow.warmup()
    role = "You are a helpful customer-service agent. Continue naturally."
    active.start_session(role, str(voice))

    rng = np.random.default_rng(42424242)
    frames = []
    inputs = []
    for seq in range(args.frames):
        # Low-amplitude deterministic speech-like broadband input exercises
        # codec state more strongly than an all-zero stream.
        pcm = (rng.standard_normal(active.frame_size) * 0.002).astype(np.float32)
        inputs.append(pcm)
        result = active.active_step(seq, pcm)
        frames.append(result.frame)
        if result.emitted_seq is not None and result.emitted_seq >= 0:
            emitted = frames[result.emitted_seq]
            emitted.played = True
            emitted.played_pcm = result.output_pcm.copy()

    memory_prompt = (
        f"{role}\n\n<conversation_memory>\n"
        "User | preferred project = Aurora\n"
        "</conversation_memory>"
    )
    shadow.start_session(memory_prompt, str(voice))
    replay = shadow.replay(frames[args.cutoff + 1 :])

    next_pcm = (rng.standard_normal(active.frame_size) * 0.002).astype(np.float32)
    active_next = active.active_step(args.frames, next_pcm)
    shadow_next = shadow.active_step(args.frames, next_pcm)
    next_user_codes_equal = bool(
        np.array_equal(active_next.frame.user_codes, shadow_next.frame.user_codes)
    )
    if active_next.output_pcm is None or shadow_next.output_pcm is None:
        next_output_mse = None
    else:
        next_output_mse = float(
            np.mean(np.square(active_next.output_pcm - shadow_next.output_pcm))
        )
    codec_match_run = (
        replay.frames
        if replay.last_user_mismatch_seq is None
        else replay.end_seq - replay.last_user_mismatch_seq
    )
    decoder_match_run = (
        replay.decoder_frames_compared
        if replay.last_decoder_mismatch_seq is None
        else int(replay.last_decoder_compared_seq) - replay.last_decoder_mismatch_seq
    )

    report = {
        "frames": args.frames,
        "cutoff": args.cutoff,
        "tail_frames": replay.frames,
        "user_code_mismatches": replay.user_code_mismatches,
        "last_user_mismatch_seq": replay.last_user_mismatch_seq,
        "codec_match_run": codec_match_run,
        "target_mismatches": replay.target_mismatches,
        "decoder_frames_compared": replay.decoder_frames_compared,
        "decoder_mismatches": replay.decoder_mismatches,
        "last_decoder_mismatch_seq": replay.last_decoder_mismatch_seq,
        "decoder_match_run": decoder_match_run,
        "next_user_codes_equal": next_user_codes_equal,
        "next_output_mse": next_output_mse,
        "decoder_mse_threshold": args.decoder_mse_threshold,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if replay.target_mismatches:
        raise RuntimeError("forced target replay diverged")
    if not next_user_codes_equal or codec_match_run < 16:
        raise RuntimeError("shadow Mimi encoder had not converged")
    if decoder_match_run < 16:
        raise RuntimeError("shadow Mimi decoder had not converged")
    if next_output_mse is None or next_output_mse > args.decoder_mse_threshold:
        raise RuntimeError("first post-replay output is discontinuous")


if __name__ == "__main__":
    with torch.no_grad():
        main()
