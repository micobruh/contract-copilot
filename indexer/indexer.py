from sentence_transformers import SentenceTransformer

# Load BGE-m3
embedding_model = SentenceTransformer("BAAI/bge-m3")

# Packaging embedding function that fits Chroma
class ChromaEmbeddingFunction:
   def __init__(self, model):
       self.model = model

   def __call__(self, input):
       return self.model.encode(input).tolist()
       
BGE_embedding_fn = ChromaEmbeddingFunction(embedding_model)

import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
client = chromadb.Client()

# Create collection
try:
    collection = client.get_collection(name="Test_RAG", embedding_function=BGE_embedding_fn)
    print("Collection exists, using existing one")
except:
    collection = client.create_collection(name="Test_RAG", embedding_function=BGE_embedding_fn)
    print("Collection does not exist, creating new one")

def store_documents_in_batches(collection, chunks, embedding_function, batch_size=100):
    """
    Load the documents in batches into ChromaDB, using a custom embedding function
    
    Args:
        collection: ChromaDB collection object
        chunks: List of document chunks
        embedding_function: Custom embedding function
        batch_size: Number of documents to process in each batch
    """
    total_chunks = len(chunks)
    successful_batches = 0
    failed_batches = []
    
    print(f"Starting to process {total_chunks} chunks, batch size: {batch_size}")
    print(f"Using custom embedding model: {type(embedding_function.model).__name__}")
    
    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:batch_end]
        
        try:
            # Extract data for the current batch
            documents = [chunk['text'] for chunk in batch_chunks]
            metadatas = [chunk['metadata'] for chunk in batch_chunks]
            ids = [chunk['id'] for chunk in batch_chunks]
            
            # Use custom embedding function to generate embeddings
            print(f"Generating embeddings for batch {successful_batches + 1}...")
            embeddings = embedding_function(documents)
            
            # Store in ChromaDB
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings  # Add custom embeddings
            )
            
            successful_batches += 1
            print(f"Batch {successful_batches}: Processed {batch_end}/{total_chunks} ({batch_end/total_chunks*100:.1f}%)")
            
            # Release memory
            del documents, metadatas, ids, embeddings
            
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {str(e)}")
            failed_batches.append((i, batch_end, str(e)))
            continue
    
    print(f"\nProcessing complete:")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {len(failed_batches)}")
    
    if failed_batches:
        print("Failed batch details:")
        for start, end, error in failed_batches:
            print(f"  Range {start}-{end}: {error}")
    
    return successful_batches, failed_batches    