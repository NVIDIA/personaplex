# Seamless-install latency analysis

This report records the 2026-09-01 measurements behind the process-isolated
PersonaPlex gateway and its subsequent batched-prefill update. **The active
audio path remains within its 80 ms p95 frame budget during installation, and
batched `M+W` preparation is substantially faster; reset-Mimi state convergence
is now the dominant readiness bottleneck.**

Raw summary values are in
[`benchmarks/seamless_latency_2026-09-01.json`](benchmarks/seamless_latency_2026-09-01.json).

> **Interpretation update.** These measurements use the branch's sequential
> teacher-forced tail path. They validate active/shadow process isolation and a
> cheap atomic handoff, but they are not measurements of the intended batched
> `[role, M, W]` prefill. The target design and migration boundary are recorded
> in [`adr/0001-active-shadow-seamless-install.md`](adr/0001-active-shadow-seamless-install.md).

## Batched-prefill update

The `M+W` path is now implemented. New raw summaries are in
[`benchmarks/batched_prefill_2026-09-01.json`](benchmarks/batched_prefill_2026-09-01.json).

| suffix | sequential suffix | batched suffix | suffix speedup | sequential total | batched total | total speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 frames | 4.11 s | 1.04 s | 3.96× | 10.64 s | 5.23 s | 2.03× |
| 256 frames | 16.06 s | 3.92 s | 4.10× | 22.58 s | 8.17 s | 2.76× |

Both process-isolated comparisons had zero user-code, forced-target, and
decoder mismatches. With the same next-step seed, assistant targets were equal
and output PCM MSE was 0.

The live controller completed its swap in 15.42 s. The prefix took 3.13 s,
duplex prefill and codec convergence took 9.29 s, the final delta was one frame,
and the critical section was 59.1 ms. Active p95 during installation was
64.49 ms. The remaining delay is dominated by reset-Mimi encoder washout, not
Transformer prefill. The sequential replay-rate model below is retained only
as the historical baseline.

The runtime now exposes exactly two architectures: two-PersonaPlex-GPU
`shadow` and one-PersonaPlex-GPU `inplace`. A first A40 system validation is in
[`benchmarks/install_backends_2026-09-01.json`](benchmarks/install_backends_2026-09-01.json).

| backend | PP GPUs | trigger to install | blocking interruption | prefix | duplex prefill | final codec convergence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `shadow` | 2 | 15.42 s | 59.1 ms | 3.13 s | 9.29 s | yes, 16 frames |
| `inplace` | 1 | 27.93 s | 7.53 s | 3.15 s | 4.38 s | no, 0 frames |

The inplace run accumulated a 256-frame (20.48 s) raw tail before reset,
installed memory successfully, and completed frame 303 after resuming. Its
decoder matched for 139 frames, but its reset Mimi encoder did not match at the
final boundary. This is why `inplace` is a resource-efficient baseline, not a
seamless system. These two validation runs used different `W` lengths, so their
prefill times are not a controlled speed comparison. Peak VRAM, perceptual
audio discontinuity, and a fixed-`W` head-to-head run remain outstanding. See
[`adr/0002-two-install-backends.md`](adr/0002-two-install-backends.md).

## Measurement scope

- Two NVIDIA A40 48 GB GPUs on one `vram48` node.
- Official PersonaPlex checkpoint at 12.5 codec frames/s (80 ms/frame).
- Real 24 kHz service sample: `assets/test/input_service.wav`.
- Active and shadow models in separate OS processes, with disjoint physical
  CPU-core groups and one PyTorch intra-op thread per process.
- A manual compact memory was used for the live controller run so that model
  installation could be measured separately from writing.
- Previously generated GPU-1 artifacts provide 24 actual Qwen compactor
  timings. Their combination with the new gateway measurement is an estimate,
  not yet a three-GPU end-to-end measurement.

The active latency starts with decoded PCM and ends when PersonaPlex returns a
codec-frame result. Browser jitter, Opus encode/decode, and network latency are
not included.

## Active-path result

| condition | frames | mean | p95 | p99 | max | >80 ms |
|---|---:|---:|---:|---:|---:|---:|
| process baseline | 48 | 65.18 ms | 67.69 ms | 69.60 ms | 69.91 ms | 0% |
| process during install | 400 | 65.19 ms | 67.75 ms | 69.82 ms | 112.86 ms | 0.25% |
| threaded during replay (ablation) | 334 | 72.87 ms | 85.37 ms | 119.42 ms | 203.26 ms | 8.98% |

The process runtime preserves the normal active distribution: the p95 change
during install is only 0.05 ms. One 112.86 ms outlier remains, so this is not a
hard real-time guarantee. A longer soak test must bound the tail before an SLO
claim.

The threaded runtime is not acceptable. It caused concurrent CUDA-graph/device
context problems, pushed p95 over the codec deadline, and failed to satisfy the
swap condition. Consequently, `moshi-seamless-server` now uses one process per
PersonaPlex GPU by default; `--slot-runtime thread` remains only for debugging.

## Startup and one manual installation

| phase | measured wall time |
|---|---:|
| load + warm both resident model processes | 26.22 s |
| cold session start, both role/voice prefixes | 3.70 s |
| installed-memory prefill while active runs | 6.75 s |
| growing-tail bulk replay | 25.16 s |
| final replay + pointer-swap critical section | 63.28 ms |
| manual trigger to completed swap | **31.98 s** |

The swap completed without reconnecting the WebSocket. Exact forced target
replay remained valid, and at readiness the codec and decoder match runs were
26 and 242 frames, respectively. The important distinction is that the atomic
route change is cheap; almost all time is spent preparing a trustworthy shadow
state.

Process isolation also passed a 128-frame cross-GPU replay check with zero user
code, forced-target, or decoder mismatches. The same check failed in the shared
process, which confirms that isolation is a correctness requirement here, not
only a latency optimization.

## Why the current sequential reference amplifies every upstream second

Let:

- `f = 12.5` be the live input rate in frames/s,
- `R = 15.898` be the measured effective replay rate while the active process
  continues,
- `C` be compactor latency, and
- `P` be installed-prefix latency.

Replay begins with `f(C+P)` queued frames. Since the tail continues to grow,
the net drain rate is only `R-f = 3.398` frames/s. The current pipeline therefore
has

```text
T_ready = (C + P) + f(C + P)/(R - f)
        = (C + P) R/(R - f).
```

For the manual run, `C=0` predicts 31.58 s, close to the observed 31.98 s. This
agreement makes the model useful for provisional capacity planning of the
historical implementation only. The subsequently measured batched path is
reported above; this replay-rate equation must not be carried over to it.

The prior 24-condition compactor artifact has mean 6.325 s and p95 6.743 s.
Combining it with this gateway gives:

| compactor assumption | estimated trigger-to-ready | context consumed | rounded reserve |
|---|---:|---:|---:|
| mean `C` | 61.17 s | 764.6 frames | 800 frames |
| p95 `C` | 63.13 s | 789.1 frames | 800 frames |

A **900-frame (72 s) engineering reserve** is a reasonable next experimental
setting, not a production SLO. The present 1,400-frame default is safe but
over-conservative. It should remain until repeated three-GPU measurements
include ASR lag, prompt-length variance, and long-session tails.

Because `R/(R-f) = 4.68`, one extra second of ASR lag, compaction, or prefill
currently adds about 4.68 seconds to readiness. Improving replay headroom is
therefore more valuable than shaving the same absolute time from one upstream
component.

## GPU-1 and transport

The local `HttpMemoryWorker` path is not the bottleneck. For a 614-event burst,
enqueue p99 was 0.015 ms and loopback drain throughput was 4,044 events/s,
versus 12.5 user audio frames/s. The reported 151.8 ms burst drain includes a
no-op compact endpoint, not Qwen generation.

Prior actual artifacts across four domains and 24 trigger conditions show:

| GPU-1 component | mean | p50 | p95 | max | recurrence |
|---|---:|---:|---:|---:|---|
| ASR model load | 11.15 s | 9.35 s | 21.00 s | 21.55 s | service startup |
| offline clip decode | 4.10 s | 4.10 s | 6.03 s | 6.25 s | not directly applicable to continuous ASR |
| compact generation | 6.325 s | 6.385 s | 6.743 s | 7.158 s | every trigger |

In deployment, streaming ASR is expected to run continuously. Its cold load is
not part of recurring trigger latency, and full-clip offline decode must not be
blindly added either. The missing number is **ASR committed-watermark lag at the
trigger**; it must be measured in the real GPU-1 service.

## Main bottlenecks and required changes

1. **Do not return to a shared CUDA process.** Separate model processes remove
   active deadline regressions and CUDA-graph contamination.
2. **Add pre-cutoff codec state restoration.** The shadow currently advances
   Mimi only from the semantic compact cutoff, so it must wait for enough new
   audio to wash out the discarded causal codec state. A state-only acoustic
   pre-roll, or an explicit encoder/decoder state snapshot, should be prepared
   without reintroducing pre-cutoff facts into the LM context.
3. **Optimize the implemented batched duplex prefill.** The 7B Transformer now
   advances over sequence chunks while LMGen's token ring remains aligned.
   Mimi remains one-frame stepped because its CUDA graphs have fixed shapes.
   Test codec-state snapshots or a state-only acoustic pre-roll, then consider
   shape-specific batched Mimi graphs.
4. **Measure the actual third GPU.** Implement and deploy streaming ASR plus the
   compactor contract, then report ASR watermark lag, compaction p95, and true
   trigger-to-swap together.
5. **Run repeated end-to-end soaks.** Include Opus/WebSocket timing, different
   voices and prompt sizes, interruptions/overlap, repeated ping-pong installs,
   audio-boundary discontinuity, memory correctness, and unchanged full-duplex
   behavior.

The current result supports the architecture decision—persistent gateway plus
process-isolated active/shadow models—but not yet a claim that installation is
fast or behaviorally seamless.
