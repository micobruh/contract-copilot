# contract-copilot

## CUAD Dataset

This project uses the official [Contract Understanding Atticus Dataset (CUAD) v1](https://huggingface.co/datasets/theatticusproject/cuad). Download it into the path expected by the indexer:

```bash
python -m pip install -U huggingface_hub
hf download theatticusproject/cuad \
  --repo-type dataset \
  --include "CUAD_v1/**" \
  --local-dir data/raw
```

The corpus will be available at `data/raw/CUAD_v1/full_contract_pdf`. The repository's `/data` directory is gitignored.

Optional alternative benchmark: https://huggingface.co/datasets/isaacus/legal-rag-bench

environment name: torch201

Potential unit type to add: 

unit_type ∈ {
  preamble,
  clause,
  subclause,
  list_item_group,
  appendix_heading,
  appendix_table_row,
  signature_or_acceptance,
  artifact
}

## Docker

Build and run the Streamlit app with Docker:

```bash
docker build -t contract-copilot .
docker run -p 8501:8501 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v /home/marco/shared_models:/app/shared_models \
  -e EMBEDDING_MODELS_ROOT=/app/shared_models/embedding_models \
  -e RERANKER_MODELS_ROOT=/app/shared_models/reranker_models \
  contract-copilot
```

Run the app together with Ollama:

```bash
docker compose up --build
```

The app container uses `OLLAMA_HOST=http://ollama:11434` to reach the Ollama service.
The Qdrant index is stored locally inside the app container under `/app/artifacts/qdrant_db`
and is persisted through the `./artifacts:/app/artifacts` volume mount, so no separate
Qdrant container is required for this setup.

## Shared Local Embedding And Reranker Models

Embedding and reranker models are stored in a shared directory so all projects under `/home/marco`
can reuse the same downloads:

```text
/home/marco/shared_models/
  embedding_models/
  reranker_models/
```

The project already points to that shared location. The existing `./local_models` path in this repo
is now just a symlink to `/home/marco/shared_models` for backward compatibility.

Before the first Docker run, install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
```

Then download the required local models with one command:

```bash
./scripts/download_models.sh
```

If a model is gated, authenticate first:

```bash
huggingface-cli login
```

The script creates `/home/marco/shared_models` by default and downloads:

- `BAAI/bge-m3`
- `jinaai/jina-embeddings-v5-text-small`
- `Qwen/Qwen3-Embedding-4B`
- `BAAI/bge-reranker-base`

If you want a different shared location, override it like this:

```bash
MODELS_ROOT=/some/other/path ./scripts/download_models.sh
```

After the download, this app and other projects can load the same files without keeping duplicate copies in each repository.

This step is required before the first `docker compose up` because the app expects these local models to already exist.

Use the command above as the CPU-safe default. It will still run on machines with no GPU.

## Faster CPU-Only Ollama Models

If you only have CPU access, smaller Ollama models will feel much faster than reasoning-heavy models.
For this project, a good default is to use `phi3` or `qwen` as the everyday model, and only use
`deepseek-r1:1.5b` when you need stronger reasoning and can tolerate slower responses.

Create a folder for local Ollama model profiles:

```bash
mkdir -p /home/marco/ollama-modelfiles
```

Create these `Modelfile`s:

```text
# /home/marco/ollama-modelfiles/phi3-fast.Modelfile
FROM phi3:latest
PARAMETER num_thread 6
PARAMETER num_ctx 2048
```

```text
# /home/marco/ollama-modelfiles/qwen-fast.Modelfile
FROM qwen:latest
PARAMETER num_thread 6
PARAMETER num_ctx 2048
```

```text
# /home/marco/ollama-modelfiles/deepseek-r1-1.5b-tuned.Modelfile
FROM deepseek-r1:1.5b
PARAMETER num_thread 6
PARAMETER num_ctx 1024
```

Build the tuned models:

```bash
ollama create phi3-fast -f /home/marco/ollama-modelfiles/phi3-fast.Modelfile
ollama create qwen-fast -f /home/marco/ollama-modelfiles/qwen-fast.Modelfile
ollama create deepseek-r1-1.5b-tuned -f /home/marco/ollama-modelfiles/deepseek-r1-1.5b-tuned.Modelfile
```

Then run one of the smaller models in Ollama and point the app to that model in your project configuration.
For CPU-only laptops, start with `qwen-fast` or `phi3-fast`. Keep `deepseek-r1-1.5b-tuned` as a fallback
for harder prompts because it will usually be slower.

If you have an NVIDIA GPU and NVIDIA Container Toolkit installed, use the GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

This enables GPU access for both:
- the Streamlit app container, so embeddings and reranking can use CUDA
- the Ollama container, so LLM generation can use the GPU

Then open `http://localhost:8501`.

## GPU Troubleshooting

Check that the host can see the GPU:

```bash
nvidia-smi
```

Check that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If the first command fails, the issue is on the host side.
If the second command fails, Docker or NVIDIA Container Toolkit is not configured correctly.

If Docker starts but the app still behaves like CPU-only:
- verify you launched with the GPU override file
- verify Ollama is running with GPU access
- check container logs with `docker compose logs -f`
