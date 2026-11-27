#!/bin/bash

set -euo pipefail

if [[ ! -d "$MODEL_PATH" || -z "$MODEL_PATH" ]]; then
    echo "Model directory does not exist: $MODEL_PATH" >&2
    exit 2
fi

OMNI_VERSION=${OMNI_VERSION:-"optimized"}
MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd -P)

DEPLOY_CONFIG="/opt/resources/personaplex_96.yaml"
IMAGE_TAG="blogpost-bench-$OMNI_VERSION:vllm-omni-0.33.0rc1"
CONTAINER_NAME="blogpost-bench-container"
WEBSERVER_PORT="8002"

echo "Starting $OMNI_VERSION $MODEL_PATH vllm-omni instance..."

if [[ "$OMNI_VERSION" == "baseline" ]]; then
    EXTRA_ARGS=()
else
    EXTRA_ARGS=(--duplex-engine-backend-as-process)
fi

docker run \
    -e HF_TOKEN \
    -e HF_HOME=/root/.cache/huggingface \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$MODEL_PATH:/root/personaplex-7b-v1" \
    --gpus all \
    --rm \
    --shm-size=16g \
    --name "$CONTAINER_NAME" \
    --entrypoint vllm \
    "$IMAGE_TAG" \
    serve "/root/personaplex-7b-v1" --omni \
    --served-model-name nvidia/personaplex-7b-v1 \
    --deploy-config "$DEPLOY_CONFIG" \
    --disable-log-stats \
    --no-collect-stage-metrics \
    --port "$WEBSERVER_PORT" "${EXTRA_ARGS[@]}"
