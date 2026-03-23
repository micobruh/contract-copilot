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

Then open `http://localhost:8501`.
