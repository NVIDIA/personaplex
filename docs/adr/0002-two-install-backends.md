# ADR 0002: Expose exactly two memory-install backends

- Status: Accepted
- Date: 2026-09-01

## Context

Memory writing and memory installation are separate concerns. Triggering and
compaction produce cumulative memory `M` at an exact cutoff `C`; the installer
must reconstruct the replacement PersonaPlex state from `Role+M` and the known
duplex tail `W=(C,t]`. Batched prefill accelerates this reconstruction, but it
does not remove contention with 12.5 Hz live full-duplex inference when both
jobs use the same GPU.

## Decision

The runtime accepts exactly two values for `--install-backend`:

1. `shadow`: two resident PersonaPlex slots on distinct GPUs. The inactive
   slot prefills `Role+M+W`, catches `ΔW`, verifies semantic and codec state,
   and atomically becomes active. Live generation continues during preparation.
2. `inplace`: one resident PersonaPlex slot. Compaction remains asynchronous,
   but the controller pauses live inference at a safe output boundary and
   destructively prefills `Role+M+W` into the active slot. It then resumes on
   the installed state. Before destructive reset it accumulates a configurable
   raw-tail reserve (`inplace_min_tail_frames`, default 256) so reset Mimi state
   has more observed audio to wash out. Codec and decoder convergence are
   reported as diagnostics rather than success gates for this baseline.

The optional ASR/compactor GPU is outside this count. Both modes use the same
memory-worker contract, cutoff validation, duplex ledger, forced targets,
batch-size controls, and convergence checks.

## Operational contract

| Property | `shadow` | `inplace` |
| --- | --- | --- |
| PersonaPlex slots | exactly 2 | exactly 1 |
| Live inference during `M+W` prefill | yes | no |
| Critical section | final `ΔW` + pointer update | complete `Role+M+W` prefill |
| Successful handoff | active pointer changes | active state changes in place |
| Failure after prefill begins | old active remains valid | session closes fail-stop |
| Codec/decoder convergence | required before swap | reported; discontinuity allowed |
| Primary latency metric | switch pause / active deadline misses | `interruption_s` |
| Intended claim | seamless installation candidate | resource-efficient baseline |

Status exposes `install_backend` and a backend-neutral `last_install`. The
compatibility field `last_swap` is populated only for `shadow`.

## Consequences

- The comparison isolates architecture: experiments must hold `M`, `W`, cutoff,
  batch size, model, and hardware generation constant.
- `shadow` pays for a second resident model replica to protect full-duplex
  deadlines.
- `inplace` provides a deployable one-PersonaPlex-GPU option and directly
  measures why batched prefill alone is not seamless.
- `inplace` may resume with a different Mimi encoder state even when the forced
  LM targets are exact; this is an evaluated limitation, not hidden failure.
- A partially failed inplace reconstruction cannot safely resume the old cache;
  the WebSocket session is closed instead of using corrupt state.

Shared-weight dual-cache scheduling, cache offload/restore, more than two
PersonaPlex replicas, and an unbatched replay mode are not runtime backends.
They may be studied as future optimizations or experimental oracles, but the
production CLI rejects any backend other than `shadow` and `inplace`.

## Required evaluation

Compare trigger-to-install time, `Role+M` and `W` prefill time, effective
prefill throughput, active-frame p50/p95/p99, 80 ms deadline misses, audible
pause, output discontinuity, peak VRAM, GPU count, exact forced-target
agreement, Mimi boundary convergence, memory use, and preservation of
full-duplex behavior. Only `shadow` can support a seamlessness claim.
