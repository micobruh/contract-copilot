import tqdm
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import chromadb
from .ocr_loader import load_corpus

from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


class UniversalEmbeddingModel:
    def __init__(
        self,
        model_name,
        device=None,
        torch_dtype=None,
        models_root="embedding_models",
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if self.device == "cuda" else torch.float32
        )

        # Map Hugging Face model names to local folder names
        self.local_model_map = {
            "BAAI/bge-m3": "bge-m3",
            "Qwen/Qwen3-Embedding-4B": "Qwen3-Embedding-4B",
            "jinaai/jina-embeddings-v5-text-small": "jina-embeddings-v5-text-small",
        }

        if model_name not in self.local_model_map:
            raise ValueError(f"Unsupported model: {model_name}")

        self.models_root = Path(models_root)
        self.model_path = self.models_root / self.local_model_map[model_name]

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local model folder not found: {self.model_path.resolve()}"
            )

        # Convert to string because HF/SentenceTransformers expect str/path-like
        model_path_str = str(self.model_path)

        if model_name == "BAAI/bge-m3":
            self.backend = "sentence_transformer"
            self.model = SentenceTransformer(
                model_path_str,
                device=self.device,
                local_files_only=True,
            )

        elif model_name == "Qwen/Qwen3-Embedding-4B":
            self.backend = "sentence_transformer"
            self.model = SentenceTransformer(
                model_path_str,
                device=self.device,
                local_files_only=True,
                model_kwargs={"torch_dtype": self.torch_dtype},
                tokenizer_kwargs={"padding_side": "left"},
            )

        elif model_name == "jinaai/jina-embeddings-v5-text-small":
            self.backend = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if self.backend == "sentence_transformer":
            if self.model_name == "Qwen/Qwen3-Embedding-4B":
                return self.model.encode(texts, prompt_name="query")
            return self.model.encode(texts)

        elif self.backend == "transformers":
            with torch.no_grad():
                return self.model.encode(
                    texts=texts,
                    task="retrieval",
                    prompt_name="document",
                )


class ChromaEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        embeddings = self.model.encode(input)
        return embeddings.tolist()


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
    print(f"Using custom embedding model: {embedding_function.model.model_name}")
    
    for i in tqdm(range(0, total_chunks, batch_size, desc="Batches Processed")):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i: batch_end]
        
        try:
            # Extract data for the current batch
            source = "Judicial College of Victoria's Criminal Charge Book"
            documents = [chunk['text'] for chunk in batch_chunks]
            metadatas = [{'source': source, 'title': chunk['title'], 'footnotes': chunk['footnotes']} for chunk in batch_chunks]
            ids = [chunk['id'] for chunk in batch_chunks]
            
            # Use custom embedding function to generate embeddings
            embeddings = embedding_function(documents)
            
            # Store in ChromaDB
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings  # Add custom embeddings
            )
            
            successful_batches += 1
            
            # Release memory
            del documents, metadatas, ids, embeddings
            
        except Exception as e:
            print(f"Batch {i // batch_size + 1} failed: {str(e)}")
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


def build_chroma_database(model_name="jinaai/jina-embeddings-v5-text-small", batch_size=10):
    # Load passages in:
    # isaacus/legal-rag-bench (Judicial College of Victoria’s Criminal Charge Book)
    legal_corpus_chunks = load_corpus()
    print("Loaded legal corpus chunks, total chunks:", len(legal_corpus_chunks))

    # Choose any model here
    # Possible models: "BAAI/bge-m3", "Qwen/Qwen3-Embedding-4B", "jinaai/jina-embeddings-v5-text-small"
    embedding_model = UniversalEmbeddingModel(model_name)
    embedding_fn = ChromaEmbeddingFunction(embedding_model)
    print(f"Initialized embedding model: {model_name} on device: {embedding_model.device}")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create collection
    try:
        collection = client.get_collection(name="Law_RAG", embedding_function=embedding_fn)
        print("Collection exists, using existing one")
    except:
        collection = client.create_collection(name="Law_RAG", embedding_function=embedding_fn)
        print("Collection does not exist, creating new one")

    successful, failed = store_documents_in_batches(
        collection=collection, 
        chunks=legal_corpus_chunks, 
        embedding_function=embedding_fn,
        batch_size=batch_size
    )

    return successful, failed