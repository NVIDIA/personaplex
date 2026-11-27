#!/bin/bash

set -euo pipefail

AUDIO_PATH=${AUDIO_PATH:-"/opt/resources/bill_boerst_looped_2m.flac"}
CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS:-"1,2,4,8,16,32,48,64,72,96"}
OUTPUT_JSONL=${OUTPUT_JSONL:-"/tmp/personaplex-results.jsonl"}

CONTAINER_NAME="blogpost-bench-container"
CONTAINER_OUTPUT_JSONL="/tmp/personaplex-results.jsonl"

docker exec -it $CONTAINER_NAME python3 /opt/resources/personaplex_benchmark.py \
    --audio "$AUDIO_PATH" \
    --concurrency-levels "$CONCURRENCY_LEVELS" \
    --output-jsonl "$CONTAINER_OUTPUT_JSONL"

docker cp "$CONTAINER_NAME:$CONTAINER_OUTPUT_JSONL" "$OUTPUT_JSONL"
