from indexer.indexer import build_chroma_database
from retriever.retriever import answer_question

# model_name = "Qwen/Qwen3-Embedding-4B"
# model_name = "jinaai/jina-embeddings-v5-text-small"
embedding_model_name = "BAAI/bge-m3"
reranker_model_name = "BAAI/bge-reranker-base"
# Build Chroma database with the specified embedding model and batch size.
# successful, failed = build_chroma_database(embedding_model_name=embedding_model_name)

query = '“There was an agreement between the accused and the deceased to seek the _____ of all parties to the agreement.” What is the missing word, and where is this specific phrase used?'
answer_question(query)