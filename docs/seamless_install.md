# Asynchronous seamless memory installation

This branch adds a production-shaped, three-GPU runtime to the official
PersonaPlex WebDemo without reconnecting the browser:

```text
browser WebSocket ──> persistent gateway ──> slot A / GPU 2 (active)
                            │                slot B / GPU 3 (shadow)
                            └── PCM ───────> ASR + compactor / GPU 1
```

The two PersonaPlex checkpoints are loaded once and remain resident. A compact
trigger never unloads or blocks the active checkpoint. GPU 1 builds a cumulative
memory at an exact committed frame cutoff; the inactive PersonaPlex receives
that memory as a new system prefix, replays the post-cutoff duplex tail, catches
up, and becomes active by an in-process pointer swap. The former active slot is
then the shadow for the next installation.

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
3. **Exact duplex replay.** The ledger stores each user PCM frame, its eight
   active Mimi codes, and the active model's target-position text plus eight
   assistant codes. The shadow teacher-forces the target cache directly. Using
   `LMGen.step(..., moshi_tokens=...)` here would be incorrect because that API
   applies codebook delays again.
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

Thus **when** to compact can change without changing the writer, replay, or
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
shadow replay RTF, catch-up lag, switch pause, audio discontinuity, exact target
replay mismatches, Mimi convergence length, memory correctness, and unchanged
full-duplex behavior before treating the installation as seamless.
