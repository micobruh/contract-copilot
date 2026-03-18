from indexer.indexer import build_chroma_database, UniversalEmbeddingModel, LocalEmbeddingWrapper
from retriever.retriever import retrieve

# model_name = "Qwen/Qwen3-Embedding-4B"
# model_name = "jinaai/jina-embeddings-v5-text-small"
model_name = "BAAI/bge-m3"
# Build Chroma database with the specified embedding model and batch size.
# successful, failed = build_chroma_database(model_name=model_name)

query = '“There was an agreement between the accused and the deceased to seek the _____ of all parties to the agreement.” What is the missing word, and where is this specific phrase used?'
docs = retrieve(query, model_name=model_name)
for doc in docs:
    print(f"Title: {doc.metadata['title']}\n")
    print(f"Text: {doc.page_content}\n")
    print("\n---\n")