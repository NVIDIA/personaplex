.DEFAULT_GOAL := all

DOCKER_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
IMAGE_TAG := vllm-omni-0.33.0rc1
IMAGE_LABEL := blogpost-bench

.PHONY: all optimized baseline clean

all: optimized baseline

optimized:
	docker build --label $(IMAGE_LABEL) -f "$(DOCKER_DIR)dockerfile.optimized" -t blogpost-bench-optimized:$(IMAGE_TAG) "$(DOCKER_DIR)"

baseline:
	docker build --label $(IMAGE_LABEL) -f "$(DOCKER_DIR)dockerfile.baseline" -t blogpost-bench-baseline:$(IMAGE_TAG) "$(DOCKER_DIR)"

# Prune benchmark images unused by any container, including tagged images.
clean:
	docker image prune --all --force --filter "label=$(IMAGE_LABEL)"
