import argparse

from .config import config
from .indexer import build_qdrant_database
from .model.llm import answer_question


def build_index_main() -> int:
    parser = argparse.ArgumentParser(description="Build the Qdrant index.")
    parser.add_argument(
        "--embedding-model",
        default=config.default_embedding_model_name,
        help="Embedding model name to use for indexing.",
    )
    args = parser.parse_args()

    successful, failed = build_qdrant_database(embedding_model_name=args.embedding_model)
    print(f"Indexing complete. Successful writes: {successful}, failed writes: {failed}")
    return 0


def run_cli_main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question through the RAG CLI.")
    parser.add_argument("query", help="Question to ask the RAG pipeline.")
    parser.add_argument(
        "--llm-model",
        default=config.default_llm_model_name,
        help="LLM model name to use for generation.",
    )
    args = parser.parse_args()

    answer = answer_question(args.query, args.llm_model)
    print(answer)
    return 0
