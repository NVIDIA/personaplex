# Two memory-install backends

The gateway intentionally exposes exactly two PersonaPlex installation
architectures through `--install-backend shadow|inplace`. Both consume the same
trigger, exact compact cutoff, cumulative memory `M`, and aligned duplex tail
`W`. They differ only in where `Role+M+W` is reconstructed and whether live
generation can continue during that reconstruction.

| Backend | PersonaPlex GPUs | Install behavior | Intended role |
| --- | ---: | --- | --- |
| `shadow` | 2 | Rebuild on a resident shadow, catch up `ΔW`, then atomically switch | Main seamless system |
| `inplace` | 1 | Pause the active slot, destructively rebuild it with batched `M+W`, then resume | Cost-efficient system / systems baseline |

The optional streaming-ASR + compactor service is a separate memory plane. If
it uses a dedicated GPU, total GPU counts are three for `shadow` and two for
`inplace`; the install comparison itself is two versus one PersonaPlex GPU.

```text
                                 ┌─ shadow: active PP GPU + shadow PP GPU
browser ── persistent gateway ─┤
          │                      └─ inplace: one active PP GPU
          └─ PCM ──> optional ASR + compactor memory plane
```

In `shadow` mode, both PersonaPlex checkpoints remain resident in separate OS
processes. A trigger never unloads or blocks the active checkpoint. The shadow
prepares the replacement state, catches the small delta created during
preparation, and becomes active by an in-process pointer swap. The former
active becomes the next shadow.

In `inplace` mode, compaction is still asynchronous, so the active model keeps
responding while `M` is written. Once the memory and a safe audio boundary are
available, the controller holds the live-step lock while the sole PersonaPlex
slot performs batched `Role+M+W` prefill. This interval is reported as
`interruption_s`; it is intentionally not described as seamless. Because the
operation replaces the only live cache, a failure after reset closes the
session instead of continuing from a partially rebuilt state.

The data plane treats `[role, M, W]` as one logical batched prefill. Role and
memory text use chunked aligned prefill, while `W` uses delay-corrected `T>1`
Transformer forwards over the exact duplex ledger. Mimi's fixed-shape CUDA
graphs still advance one codec frame at a time to restore causal boundary state;
this does not autoregressively regenerate the conversation. The decision, cost
model, implementation boundary, and batched-prefill acceptance tests are in
[`adr/0001-active-shadow-seamless-install.md`](adr/0001-active-shadow-seamless-install.md).
The two-backend runtime decision and comparison contract are in
[`adr/0002-two-install-backends.md`](adr/0002-two-install-backends.md).

This is intentionally a gateway, not a reverse proxy. Opus reader/writer state,
the browser WebSocket, frame numbering, and playback remain owned by one
process across every installation.

## Correctness invariants

1. **No compaction on the response path.** `process_pcm` only enqueues PCM for
   the memory worker. Compaction runs in a separate task in both backends.
2. **Exact cutoff.** GPU 1 must return the requested `cutoff_hint_seq` exactly.
   A different cutoff is rejected so no evidence is lost or represented in
   both compact memory and the raw tail.
3. **Exact duplex prefill.** The ledger stores each user PCM frame, its eight
   active Mimi codes, and the active model's target-position text plus eight
   assistant codes. `W` is known observed context, so the install slot
   teacher-forces it rather than sampling it. The default implementation batches the main
   Transformer while advancing the delay ring exactly. Using
   `LMGen.step(..., moshi_tokens=...)` naively would be incorrect because that
   API applies codebook delays again.
4. **Codec convergence.** Tail PCM advances the shadow Mimi encoder and the
   forced assistant codes advance its decoder. Initial states may differ
   because pre-cutoff acoustics were discarded. Shadow mode withholds its swap
   until both user codes and already-played assistant PCM match for configured
   runs. Inplace mode accumulates at least `--inplace-min-tail-frames` before it
   destructively resets the only slot, then reports convergence without making
   it a success gate; representation discontinuity is part of that baseline.
5. **Backend-specific critical section.** In `shadow`, only final delta prefill
   and pointer update hold the live-step lock, and preparation failure leaves
   the active slot untouched. In `inplace`, the complete batched prefill holds
   the lock and is measured as user-visible interruption.
6. **Audible boundary.** Both backends wait for consecutive quiet assistant
   output frames. `force_switch_after_s` is a declared fallback rather than a
   silent timing heuristic.
7. **Reserve-based trigger.** Installation starts before the context limit.
   Waiting until the cache is full cannot work because compaction and prefill
   have non-zero latency. This capacity trigger is a fallback policy,
   not a claim that the research trigger problem is solved.

## GPU-1 memory-worker API

`--memory-worker-url` points to an independently deployed streaming ASR and
compactor service. This repository does not silently substitute authored text
or browser captions for ASR. The HTTP contract is:

### `POST /v1/session/start`

```json
{"session_id": "...", "role_prompt": "..."}
```

### `POST /v1/session/audio`

The body is one mono `float32` little-endian PCM frame. Headers are:

```text
X-Session-Id: <id>
X-Frame-Seq: <monotonic codec-frame index>
X-Sample-Rate: 24000
Content-Type: audio/f32le
```

The service must acknowledge only after accepting the frame into durable
session order. Uploads use a bounded asynchronous queue. Overflow or sender
failure disables automatic installation for that session but does not stop the
active PersonaPlex conversation.

### `POST /v1/session/assistant`

PersonaPlex text pieces are already available without another ASR pass. The
gateway sends each non-padding piece with its logical (delay-corrected) frame:

```json
{"session_id": "...", "frame_seq": 1579, "text_piece": " Aurora"}
```

A caller-only memory policy may retain this as audit context without writing it
into memory. A general service policy can use it to preserve prior agent
commitments and completed actions. In either case, the source must remain
distinguishable from caller ASR; never rewrite an agent statement as a
caller-provided fact.

### `POST /v1/session/compact`

```json
{
  "session_id": "...",
  "request_id": "...",
  "parent_version": 0,
  "cutoff_hint_seq": 1592,
  "previous_memory": "...",
  "role_prompt": "...",
  "ledger_digest": "..."
}
```

The worker must wait until ASR evidence through the requested cutoff is
committed, recursively update the **complete** memory, and return:

```json
{
  "request_id": "...",
  "parent_version": 0,
  "cutoff_seq": 1592,
  "memory": "complete cumulative factual memory",
  "metadata": {"asr_committed_through_seq": 1592}
}
```

Do not include information after the cutoff. Preserve corrections, unresolved
goals, entity identity, exact identifiers, and uncertainty; the newer raw tail
will override older memory if they conflict.

### `POST /v1/session/close`

```json
{"session_id": "..."}
```

## Run

Install the package from this checkout, then launch the memory service
separately when automatic compaction is required.

### Shadow backend: two PersonaPlex GPUs

```bash
cd moshi
pip install -e .

moshi-seamless-server \
  --install-backend shadow \
  --host 0.0.0.0 \
  --port 8998 \
  --active-device cuda:1 \
  --shadow-device cuda:2 \
  --prefill-batch-frames 64 \
  --text-prefill-chunk-tokens 64 \
  --memory-worker-url http://127.0.0.1:9010 \
  --control-token "$SEAMLESS_CONTROL_TOKEN"
```

### Inplace backend: one PersonaPlex GPU

```bash
moshi-seamless-server \
  --install-backend inplace \
  --host 0.0.0.0 \
  --port 8998 \
  --active-device cuda:1 \
  --prefill-batch-frames 64 \
  --text-prefill-chunk-tokens 64 \
  --inplace-min-tail-frames 256 \
  --memory-worker-url http://127.0.0.1:9010 \
  --control-token "$SEAMLESS_CONTROL_TOKEN"
```

`--shadow-device` is ignored in `inplace` mode. No third installation backend
is accepted by either the CLI or `ControllerConfig`. The 256-frame inplace
tail is a conservative codec-state washout reserve, not semantic replay and not
a learned trigger policy. It improves the chance of codec convergence but does
not guarantee it.

`--slot-runtime process` is the default. `--slot-runtime thread` exists only as
a diagnostic ablation and should not be used for a seamless-latency claim.

The standard PersonaPlex WebDemo still connects to `/api/chat`. Runtime state
is visible at:

```bash
curl http://127.0.0.1:8998/api/seamless/status
```

Without a GPU-1 service, run in manual mode by omitting
`--memory-worker-url`, then submit an auditable compact candidate:

```bash
curl -X POST http://127.0.0.1:8998/api/seamless/install \
  -H "Authorization: Bearer $SEAMLESS_CONTROL_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"memory":"User | preferred project = Aurora"}'
```

To evaluate a heuristic or learned trigger independently of writing, add
`--disable-auto-trigger` and let the trigger process request GPU-1 compaction:

```bash
curl -X POST http://127.0.0.1:8998/api/seamless/install \
  -H "Authorization: Bearer $SEAMLESS_CONTROL_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Thus **when** to compact can change without changing the writer or either
installation backend.

## What is and is not validated

The CPU test suite covers both slot layouts, non-blocking compaction, batched
inplace prefill, exact-tail routing, ping-pong swaps, cutoff rejection, silence
boundaries, and ledger pinning:

```bash
PYTHONPATH=moshi python -m unittest discover -s moshi/tests -v
```

An A40 inplace smoke run completed a 256-frame `W` install on one PersonaPlex
GPU with a 7.53 s blocking interruption. Its decoder converged, but its reset
Mimi encoder did not match the former active state at the final boundary. The
raw result and the earlier shadow measurement are summarized in
[`benchmarks/install_backends_2026-09-01.json`](benchmarks/install_backends_2026-09-01.json).

The sequential-oracle versus batched-prefill check needs two 48-GB GPUs:

```bash
PYTHONPATH=moshi python moshi/tests/gpu_batched_prefill_benchmark.py \
  --frames 256 \
  --output batched-prefill.json
```

An end-to-end comparison must run both backends with identical `M`, `W`,
cutoff, and batch size. Report active-frame p50/p95/p99, deadline misses,
`M+W` prefill latency, effective throughput, peak VRAM, `interruption_s`, final
delta size, audio discontinuity, exact target mismatches, Mimi boundary-state
equivalence, memory correctness, and full-duplex behavior. Only `shadow` is a
candidate for a seamlessness claim; `inplace` quantifies the resource/latency
tradeoff. Replay RTF is retained only for the sequential equivalence oracle.

For a controlled 256-frame `W`, run each backend with
`--install-at-frame 303 --cutoff-seq 46`; the manual memory must describe only
evidence through frame 46.

The current A40 measurements, catch-up equation, provisional trigger reserve,
and known bottlenecks are documented in
[`seamless_latency_analysis.md`](seamless_latency_analysis.md). Reproduce the
system-level measurements with:

```bash
PYTHONPATH=moshi python moshi/tests/gpu_process_isolation_benchmark.py \
  --output process-isolation.json
PYTHONPATH=moshi python moshi/tests/gpu_process_gateway_benchmark.py \
  --install-backend shadow --output process-gateway-shadow.json
PYTHONPATH=moshi python moshi/tests/gpu_process_gateway_benchmark.py \
  --install-backend inplace --output process-gateway-inplace.json
PYTHONPATH=moshi python moshi/tests/gpu_batched_prefill_benchmark.py \
  --frames 256 --output batched-prefill.json
PYTHONPATH=moshi python moshi/tests/http_memory_latency_benchmark.py \
  --output memory-transport.json
```
