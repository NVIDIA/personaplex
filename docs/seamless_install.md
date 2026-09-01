# Asynchronous seamless memory installation

This branch adds a production-shaped, three-GPU runtime to the official
PersonaPlex WebDemo without reconnecting the browser:

```text
browser WebSocket ──> persistent gateway ──> slot A / GPU 2 (active)
                            │                slot B / GPU 3 (shadow)
                            └── PCM ───────> ASR + compactor / GPU 1
```

The two PersonaPlex checkpoints are loaded once and remain resident in separate
OS processes. Process isolation is required because a shared Python/CUDA
process caused CUDA-graph interference and active-stream deadline misses in the
two-GPU benchmark. A compact trigger never unloads or blocks the active
checkpoint. GPU 1 builds cumulative memory `M` at an exact committed frame
cutoff. The inactive PersonaPlex prepares a replacement state from the new
prefix and the known aligned duplex suffix `W`, catches the small delta created
during preparation, and becomes active by an in-process pointer swap. The
former active slot is then the shadow for the next installation.

The target data plane treats `[role, M, W]` as one logical batched prefill. The
current branch still advances `W` with teacher-forced one-frame LM steps because
upstream `LMGen` only exposes an `S=1` generation path. This is a correctness
reference rather than the intended performance optimization; it does not
autoregressively regenerate the conversation. The decision, cost model, exact
implementation boundary, and batched-prefill acceptance tests are recorded in
[`adr/0001-active-shadow-seamless-install.md`](adr/0001-active-shadow-seamless-install.md).

This is intentionally a gateway, not a reverse proxy. Opus reader/writer state,
the browser WebSocket, frame numbering, and playback remain owned by one
process across every swap.

## Correctness invariants

1. **No compaction on the response path.** `process_pcm` only enqueues PCM for
   the memory worker. Compaction, prefix prefill, and bulk replay run in a
   separate task and on separate GPUs.
2. **Exact cutoff.** GPU 1 must return the requested `cutoff_hint_seq` exactly.
   A different cutoff is rejected so no evidence is lost or represented in
   both compact memory and the raw tail.
3. **Exact duplex prefill.** The ledger stores each user PCM frame, its eight
   active Mimi codes, and the active model's target-position text plus eight
   assistant codes. `W` is known observed context, so the shadow teacher-forces
   it rather than sampling it. The current reference implementation forces each
   target cache position directly; the target implementation performs the same
   state transition as a batched sequence prefill. Using
   `LMGen.step(..., moshi_tokens=...)` naively would be incorrect because that
   API applies codebook delays again.
4. **Codec convergence.** Tail PCM advances the shadow Mimi encoder and the
   forced assistant codes advance its decoder. Initial states may differ
   because pre-cutoff acoustics were discarded; the swap is withheld until
   both user codes and already-played assistant PCM match for configured runs.
5. **Atomic route change.** Only the final catch-up and pointer update hold the
   live step lock. A stale result, mismatched replay, timeout, or GPU-1 failure
   leaves the current active slot untouched.
6. **Audible boundary.** The normal swap waits for consecutive quiet assistant
   output frames. `force_switch_after_s` is a declared fallback rather than a
   silent timing heuristic.
7. **Reserve-based trigger.** Installation starts before the context limit.
   Waiting until the cache is full cannot work because compaction, prefill, and
   replay all have non-zero latency. This capacity trigger is a fallback policy,
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

Install the package from this checkout, then launch GPU 1's memory service
separately. The gateway process uses GPUs 2 and 3 only:

```bash
cd moshi
pip install -e .

moshi-seamless-server \
  --host 0.0.0.0 \
  --port 8998 \
  --active-device cuda:1 \
  --shadow-device cuda:2 \
  --memory-worker-url http://127.0.0.1:9010 \
  --control-token "$SEAMLESS_CONTROL_TOKEN"
```

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

Thus **when** to compact can change without changing the writer, prefill, or
atomic installation mechanism.

## What is and is not validated

The CPU test suite covers non-blocking compaction, exact-tail routing,
ping-pong swaps, cutoff rejection, silence boundaries, and ledger pinning:

```bash
PYTHONPATH=moshi python -m unittest discover -s moshi/tests -v
```

The model-level replay check needs two 48-GB GPUs:

```bash
PYTHONPATH=moshi python moshi/tests/gpu_seamless_smoke.py
```

An end-to-end claim still requires a two-PersonaPlex-GPU soak test plus the
actual GPU-1 ASR/compactor service. Report at least active-frame p50/p95/p99,
`M+W` prefill latency by sequence length, effective prefill throughput, final
delta size, switch pause, audio discontinuity, exact target mismatches, Mimi
boundary-state equivalence, memory correctness, and unchanged full-duplex
behavior before treating the installation as seamless. Replay RTF remains
useful only for the current sequential reference path.

The current A40 measurements, catch-up equation, provisional trigger reserve,
and known bottlenecks are documented in
[`seamless_latency_analysis.md`](seamless_latency_analysis.md). Reproduce the
system-level measurements with:

```bash
PYTHONPATH=moshi python moshi/tests/gpu_process_isolation_benchmark.py \
  --output process-isolation.json
PYTHONPATH=moshi python moshi/tests/gpu_process_gateway_benchmark.py \
  --output process-gateway.json
PYTHONPATH=moshi python moshi/tests/http_memory_latency_benchmark.py \
  --output memory-transport.json
```
