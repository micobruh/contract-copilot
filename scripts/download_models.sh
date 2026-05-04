#!/usr/bin/env bash

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/home/marco/shared_models}"
EMBEDDING_ROOT="${MODELS_ROOT}/embedding_models"
RERANKER_ROOT="${MODELS_ROOT}/reranker_models"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required. Install it with: pip install -U huggingface_hub"
  exit 1
fi

mkdir -p "${EMBEDDING_ROOT}" "${RERANKER_ROOT}"

echo "Downloading embedding models into ${EMBEDDING_ROOT}"
huggingface-cli download BAAI/bge-m3 \
  --local-dir "${EMBEDDING_ROOT}/bge-m3"
huggingface-cli download jinaai/jina-embeddings-v5-text-small \
  --local-dir "${EMBEDDING_ROOT}/jina-embeddings-v5-text-small"
huggingface-cli download Qwen/Qwen3-Embedding-4B \
  --local-dir "${EMBEDDING_ROOT}/Qwen3-Embedding-4B"

echo "Downloading reranker models into ${RERANKER_ROOT}"
huggingface-cli download BAAI/bge-reranker-base \
  --local-dir "${RERANKER_ROOT}/bge-reranker-base"

echo "Model download complete."
