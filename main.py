from indexer.indexer import build_chroma_database, UniversalEmbeddingModel, ChromaEmbeddingFunction
import chromadb

model_name = "jinaai/jina-embeddings-v5-text-small"
# Build Chroma database with the specified embedding model and batch size.
successful, failed = build_chroma_database(model_name=model_name, batch_size=10)

# Load dataset
client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = UniversalEmbeddingModel(model_name)
embedding_fn = ChromaEmbeddingFunction(embedding_model)
collection = client.get_collection(
    name="Law_RAG",
    embedding_function=embedding_fn
)

query = '“There was an agreement between the accused and the deceased to seek the _____ of all parties to the agreement.” What is the missing word, and where is this specific phrase used?'
query_vec = embedding_model.encode(query)
results = collection.query(
    query_embeddings=[query_vec],
    n_results=3
)
print(f"Title: {results['metadatas'][0]['title']} - Text: {results['documents'][0]}")