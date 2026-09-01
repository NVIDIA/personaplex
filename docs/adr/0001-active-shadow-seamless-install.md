# ADR 0001: Active-shadow installation for PersonaPlex memory

- Status: accepted and implemented; end-to-end soak validation pending
- Date: 2026-09-01
- Scope: installing compact conversation memory without interrupting a live
  full-duplex PersonaPlex session

## Decision

Use two process-isolated PersonaPlex instances as alternating active and shadow
slots. The active slot remains the only response path while the shadow prepares
a complete replacement streaming state. Once the shadow has incorporated the
compact memory and the exact live context, the persistent gateway catches up a
bounded final delta and atomically changes the active-slot pointer.

The intended shadow preparation is **one logical prefill of `M + W`**, not
autoregressive regeneration of the conversation:

```text
                         persistent gateway
browser WebSocket ──────────────┬────────────────> active PersonaPlex / GPU A
                                │                         │
                                │ exact duplex ledger     │ keeps serving
                                │                         │
trigger ─> compact history ─> M ├─> prefill [role, M, W] ─> shadow / GPU B
                                │                         │
                                └─> prefill final delta ──┘
                                             │
                                  validate boundary state
                                             │
                                  atomic active-pointer swap
```

Here:

- `C` is the exact semantic compaction cutoff.
- `M` is the cumulative memory representing history through `C`.
- `S` is a ledger snapshot taken after compaction.
- `W = ledger(C, S]` is the aligned user-audio, assistant-text, and
  assistant-audio suffix known at snapshot `S`.
- `ΔW = ledger(S, now]` is the usually short suffix produced while the shadow
  prefill is running.

`W` is observed context. Its assistant tokens are teacher-forced; they must not
be sampled again. “Prefill” therefore means advancing the Transformer cache and
the required streaming codec state from known, aligned duplex streams.

## Why a shadow makes the install seamless

Changing an already-built causal prefix invalidates the downstream cache. An
in-place active install would expose the full `M + W` prefill latency as a pause.
The shadow does not remove that compute; it removes it from the live response
path. Only a bounded final delta, readiness validation, and route change enter
the handoff critical section.

The cost model for the target path is:

```text
T_background = T_compact
             + T_prefill(M + W_snapshot)
             + Σ T_prefill(ΔW_background)
             + T_validate

T_visible    = T_lock_wait
             + T_prefill(ΔW_final)
             + T_pointer_swap
```

This model intentionally has no frame-by-frame “tail replay” term. If the
duplex suffix is processed with a real sequence prefill, the relevant
measurements are prefill latency versus sequence length, effective prefill
throughput, final-delta size, peak KV memory, and the handoff pause.

Two inference GPUs are the architectural requirement for hiding install work:
one active and one shadow. A separate streaming-ASR/compactor GPU is an
orthogonal deployment choice, so the complete reference deployment uses three
GPUs even though the handoff mechanism itself is dual-GPU.

## What “seamless” means

A successful pointer swap alone is insufficient. A run is seamless only when
all of the following hold:

1. The browser WebSocket and Opus reader/writer remain connected.
2. No user frame is lost, duplicated, or reordered across the cutoff or swap.
3. Active inference continues to meet the 80 ms codec-frame budget while the
   shadow is prepared.
4. The first post-swap assistant audio has no perceptible gap, repetition, or
   click at the boundary.
5. User-audio, assistant-text, and assistant-audio stream positions agree.
6. The installed model can use memory `M` without degrading interruption,
   backchannel, turn-taking, or other full-duplex behavior.
7. A failed or stale install leaves the current active slot untouched.

Until these are tested end to end, the precise claim is that the architecture
**enables** seamless installation, not that every current run is already
perceptually seamless.

## Implemented state transition

1. The gateway pins the ledger at `C + 1` and asks the writer for cumulative
   memory through exactly `C`.
2. After receiving `M`, it snapshots the ledger at `S`.
3. The shadow is reset and prefills `[role, voice, M, W(C, S]]` as one logical
   multimodal prefix.
4. While the active slot continues serving, the shadow prefills newly arrived
   `ΔW` in chunks. The chunk size is an implementation parameter, not a
   semantic replay policy.
5. At a safe acoustic boundary, the gateway briefly acquires the live-step
   lock and prefills the final `ΔW`.
6. It validates the ledger cursor and codec/decoder boundary state, changes one
   active-slot pointer, and releases the ledger pin.
7. The former active slot becomes the shadow for the next installation.

The cutoff contract prevents gaps and double representation: `M` contains
information at or before `C`, while `W` begins strictly after `C`. Newer raw
context overrides an older compact fact on conflict.

## Current implementation mapping

The branch already implements the production control plane:

| responsibility | implementation |
|---|---|
| persistent browser connection, Opus state, and routing | `moshi/moshi/seamless_server.py` |
| active/shadow state machine, ledger pin, and atomic swap | `moshi/moshi/seamless/controller.py` |
| exact duplex frame ledger | `moshi/moshi/seamless/ledger.py` |
| process-isolated PersonaPlex slots | `moshi/moshi/seamless/model_slot.py` |
| streaming-ASR/compactor HTTP contract | `moshi/moshi/seamless/memory.py` |
| CPU correctness tests | `moshi/tests/test_seamless_*.py` |
| two-GPU measurements | `docs/seamless_latency_analysis.md` |

The optimized data plane is now the controller default:

```text
default implementation
  batched text prefill([role, M])
  + prefill_duplex(W_snapshot) in T>1 chunks
  + prefill_duplex(ΔW)
  + atomic swap

equivalence oracle / rollback
  sequential text steps([role, M])
  + PersonaPlexSlotEngine.replay(W)
```

`LMGen.prefill_aligned()` batches the role and memory text streams.
`LMGen.prefill_forced_targets()` builds delay-corrected Transformer inputs from
the ledger while advancing LMGen's short ring exactly as repeated forced steps
would. `PersonaPlexSlotEngine.prefill_duplex()` restores the Mimi boundary state
and calls the main 7B Transformer with `T>1`. The controller uses this method for
both the snapshot suffix and final delta. `--prefill-batch-frames` and
`--text-prefill-chunk-tokens` default to 64.

Mimi remains one-frame stepped because its production encoder/decoder CUDA
graphs are captured at a fixed codec-frame shape. This is codec-state
advancement, not autoregressive conversation replay; the expensive 7B LM is
still sequence-prefilled. `PersonaPlexSlotEngine.replay()` remains available as
the sequential equivalence oracle.

The implementation has these properties:

1. Build the delay-corrected model-input sequence from each frame's stored
   eight user codes and target-position text plus eight assistant codes.
2. Run the main causal Transformer on time chunks larger than one while its KV
   cache remains in streaming mode.
3. Advance LMGen's offset, provided-mask, and short delayed-token ring exactly
   as repeated forced steps would.
4. Advance Mimi's encoder and decoder at the same boundary with their existing
   fixed-shape CUDA graphs.
5. Return target, encoder-code, and decoded-PCM validation information.
6. Keep sequential forced steps for equivalence tests and operational rollback.

## Acceptance tests

For identical role, voice, memory, and duplex frames, compare sequential forced
steps with batched prefill at multiple chunk sizes. The implemented path has
already met the following model-level requirements at 64 and 256 frames:

- identical Transformer KV offsets and cache contents within numerical
  tolerance;
- identical next-frame text/audio token output under a fixed seed;
- zero target-position mismatches;
- equivalent Mimi encoder codes at the boundary;
- equivalent decoded boundary audio and no first-frame discontinuity;
- exact ledger cursor after a concurrently produced final delta;
- substantially lower `M + W` preparation latency than the sequential
  reference.

Repeated A→B→A soak runs, real WebSocket audio inspection, memory-use quality,
and full-duplex behavioral preservation remain required before a paper claim.

## Evidence and limitations

The measured process-isolated implementation kept active p95 at 67.75 ms during
installation against an 80 ms frame budget, with a 0.25% deadline-miss rate in
400 frames. The pointer-swap critical section was 63.3 ms. A shared-process
threaded ablation exceeded the deadline at p95 and produced CUDA/stream-state
problems, so one OS process per PersonaPlex GPU remains a correctness boundary.

New A40 process-isolated measurements are in
[`../benchmarks/batched_prefill_2026-09-01.json`](../benchmarks/batched_prefill_2026-09-01.json).
At 64 frames, batched duplex prefill was 3.96× faster than sequential forcing;
at 256 frames it was 4.10× faster. Total role-memory-plus-duplex preparation
improved by 2.03× and 2.76×, respectively. Both lengths had zero user-code,
forced-target, or decoder mismatches, and their next assistant target and PCM
matched the sequential oracle exactly.

The live gateway completed a swap in 15.42 s with one final-delta frame and a
59.1 ms critical section. Active p95 during installation was 64.49 ms. Most of
the remaining readiness time came from waiting for the reset Mimi encoder to
produce 16 consecutive codes matching the active encoder. Thus the principal
next systems task is pre-cutoff codec-state restoration, not LM tail replay.

These measurements establish the batched state transition, live-path
isolation, and cheap handoff. They do not yet establish perceptual audio
seamlessness or memory-conditioned behavioral quality.
