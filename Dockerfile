FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY app.py ./app.py
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --no-build-isolation -e .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
