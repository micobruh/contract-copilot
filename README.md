# contract-copilot

Another data source: https://huggingface.co/datasets/isaacus/legal-rag-bench
environment name: torch201

## Docker

Build and run the Streamlit app with Docker:

```bash
docker build -t contract-copilot .
docker run -p 8501:8501 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/local_models:/app/local_models \
  contract-copilot
```

Run the app together with Ollama:

```bash
docker compose up --build
```

The app container uses `OLLAMA_HOST=http://ollama:11434` to reach the Ollama service.

Use the command above as the CPU-safe default. It will still run on machines with no GPU.

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
