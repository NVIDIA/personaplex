# PersonaPlex vLLM-omni Benchmark Reproduction

The resources in this directory can be used for validating the performance impact 
of a series vLLM-omni optimizations up-streamed by NVIDIA, using the PersonaPlex
model: these correspond  to PRs [9491](https://github.com/vllm-project/vllm-omni/pull/9491) 
(thread offload optimizations), [9588](https://github.com/vllm-project/vllm-omni/pull/9588) 
(process-based orchestrator) and [9809](https://github.com/vllm-project/vllm-omni/pull/9809) 
(batch coalescing support). We use vLLM-omni `0.33.0rc1` as reference version.

## Requirements

The following requirements must be satisfied for running the benchmarks:

- Linux with Bash, GNU Make, Docker, and NVIDIA Container Toolkit configured so
  `docker run --gpus all` can access the GPU.
- The HuggingFace CLI and its dependencies to download the PersonaPlex checkpoint.
- An NVIDIA driver compatible with the CUDA 13.0 runtime, and a GPU
  with FP8 support (compute capability 8.9 or higher).
- Enough host RAM/shared memory for `--shm-size=16g`, and disk space for both
  images, build cache, and model files.

All other Python and benchmark dependencies are pulled and installed
automatically into Docker images as part of the setup process.

Note that the supplied PersonaPlex deployment profile is designed to
accommodate 96 concurrent sessions on RTX Pro 6000: it reserves 
**71 GiB for the KV cache**, plus model weights and runtime allocations; 
GPUs with memory smaller than 96GB might require changes to
[personaplex_96.yaml](docker/resources/personaplex_96.yaml).

## Model and Image Preparation

Before starting, the PersonaPlex model checkpoint should be downloaded to a local
directory:

```bash
hf download nvidia/personaplex-7b-v1 --local-dir "$HOME/models/personaplex-7b-v1"
```

The `nvidia/personaplex-7b-v1` repository is gated, and downloading it requires an 
HuggingFace access token via the `HF_TOKEN` env variable or the `hf auth login` command. 
Also note that we do not simply supply the `nvidia/personaplex-7b-v1` model tag
to vLLM-omni due to a bug in `v0.33.0rc1` that prevents correct resolution of the bundled
persona voice samples.

Once the checkpoint has been downloaded, `cd` to this directory and run the following command:

```bash
make -C docker all
```

This will trigger the build of two images, both based on `vllm/vllm:v0.33.0`:

- `blogpost-bench-baseline:vllm-omni-0.33.0rc1`: an image that has the
unmodified vLLM-omni `v0.33.0rc1` wheel installed, plus all benchmark resources.
- `blogpost-bench-optimized:vllm-omni-0.33.0rc1`: an image that has a custom
vLLM-omni `v0.33.0rc1` version installed from the source, including our optimizations
from PRs [9491](https://github.com/vllm-project/vllm-omni/pull/9491), 
[9588](https://github.com/vllm-project/vllm-omni/pull/9588) and
[9809](https://github.com/vllm-project/vllm-omni/pull/9809).

Run `make -C docker clean` to remove unused benchmark images, including tagged images.

## Start the Server

On a Linux shell, run the following command to run the optimized variant of the vLLM-omni
PersonaPlex server:

```bash
MODEL_PATH="$HOME/models/personaplex-7b-v1" OMNI_VERSION=optimized ./start_personaplex.sh
```

For the baseline, use `OMNI_VERSION=baseline` instead. The launcher uses the same
container name, `blogpost-bench-container`, for both variants, so run them
separately. The following lines signal that the webserver is ready and the 
benchmark can be launched:

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The host model directory and Hugging Face cache are mounted into the container.
An exported `HF_TOKEN` is also forwarded if present. The server listens on port
8002 inside the container; the benchmark connects from that same container.

## Run the Benchmark

In a second terminal window on the same Linux host, run the following command to
launch a benchmark in the same `blogpost-bench-container` container as the webserver:

```bash
CONCURRENCY_LEVELS="64,96" OUTPUT_JSONL=/tmp/personaplex.jsonl ./start_benchmark.sh
```

The [personaplex_benchmark.py](docker/resources/personaplex_benchmark.py) file
contains the full benchmarking logic. Each run first warms up at its largest 
concurrency, then measures two requests per worker at each level. The bundled 
audio lasts 120 seconds, so even a small sweep takes several minutes. Worker 
starts are staggered by up to five seconds.

The bundled audio sample is a looped version of the `bill_boerst.wav` sample from
the [tts-voices](https://huggingface.co/kyutai/tts-voices) library. Note that a custom 
audio file can also be supplied via the `AUDIO_PATH` env variable: this should be first 
copied into the `blogpost-bench-container` container via `docker cp`. The `OUTPUT_JSONL` 
env variable then controls the path on the host where detailed JSONL measurements are saved 
upon exit.

## Results

The output JSONL contains one record per level with configuration, aggregate distributions, 
and individual request measurements, including inter-delta intervals and failures. Statistics 
are printed to stderr after each concurrency level, like in the following example:

```text
N=96: 192 requests
First input to coverage: avg=121,531 ms std=283 ms span p16..p84=647 ms
  q=0.00: 121,075 ms
  q=0.16: 121,159 ms
  q=0.25: 121,254 ms
  q=0.50: 121,562 ms
  q=0.75: 121,720 ms
  q=0.84: 121,805 ms
  q=0.90: 121,870 ms
  q=0.99: 122,096 ms
  q=1.00: 122,112 ms
First audio delta latency: avg=297 ms std=137 ms span p16..p84=291 ms
  q=0.00: 137 ms
  q=0.16: 191 ms
  q=0.25: 203 ms
  q=0.50: 229 ms
  q=0.75: 339 ms
  q=0.84: 482 ms
  q=0.90: 530 ms
  q=0.99: 634 ms
  q=1.00: 676 ms
Inter-delta audio latency: avg=160 ms std=82 ms span p16..p84=49 ms
  q=0.00: 10 ms
  q=0.16: 127 ms
  q=0.25: 139 ms
  q=0.50: 154 ms
  q=0.75: 169 ms
  q=0.84: 176 ms
  q=0.90: 183 ms
  q=0.99: 344 ms
  q=1.00: 1,299 ms
N=96: 192/192 succeeded; stream RTF 1.0015970813565975
```

The RTF - or real-time factor - is the elapsed time from the first output audio 
delta divided by received audio duration: **lower is better**, and 1.0 or
lower values mean real-time performance. Inter-delta timings start at the
second nonempty audio delta, while the first-delta latency is measured separately. 

Input submission is paced in real time at 80ms intervals, so this benchmark measures 
streaming behavior rather than maximum offline generation speed. Completion
requires input upload and output duration coverage; it does not verify that the
model has finished responding or assess audio quality.
