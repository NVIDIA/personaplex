#!/bin/bash

REPO_URL="https://github.com/vllm-project/vllm-omni"
REPO_TAG="v0.33.0rc1"
CHECKOUT_PATH="/tmp/vllm-omni-checkout"

set -euo pipefail

git clone --branch $REPO_TAG $REPO_URL $CHECKOUT_PATH

cd $CHECKOUT_PATH

git apply /opt/resources/PR_9491_thread_offload_optimizations.patch
git apply /opt/resources/PR_9588_process_based_orchestrator.patch
git apply /opt/resources/PR_9809_batch_coalescing.patch

uv pip install --system --python python3 --no-cache-dir \
   --constraint /opt/resources/constraints.txt .