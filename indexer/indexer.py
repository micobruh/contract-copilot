from tqdm import tqdm
from pathlib import Path
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from .ocr_loader import load_corpus


class UniversalEmbeddingModel:
    def __init__(
        self,
        model_name,
        device=None,
        dtype=None,
        models_root="embedding_models",
    ):
        self.model_name = model_name

        # Force CPU if current PyTorch build cannot actually use your GPU
        if device is None:
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.dtype = dtype or (
            torch.bfloat16 if self.device == "cuda" else torch.float32
        )

        self.local_model_map = {
            "BAAI/bge-m3": "bge-m3",
            "Qwen/Qwen3-Embedding-4B": "Qwen3-Embedding-4B",
            "jinaai/jina-embeddings-v5-text-small": "jina-embeddings-v5-text-small",
        }

        if model_name not in self.local_model_map:
            raise ValueError(f"Unsupported model: {model_name}")

        self.models_root = models_root
        self.model_path = Path(f"./{self.models_root}/{self.local_model_map[model_name]}")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local model folder not found: {self.model_path.resolve()}"
            )

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
                model_kwargs={"dtype": self.dtype},
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
                dtype=self.dtype,
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
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return embeddings
    

class LocalEmbeddingWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()    


def store_documents_in_batches(chunks, embedding_wrapper, batch_size=2):
    total_chunks = len(chunks)
    successful_writes = 0
    failed_writes = 0

    print(f"Starting to process {total_chunks} chunks")
    print(f"Batch size: {batch_size}")
    print(f"Using custom embedding model: {embedding_wrapper.model.model_name}")

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "id": chunk["id"],
                "title": chunk["title"],
                "footnotes": chunk["footnotes"] if chunk["footnotes"] else "",
                "source": "Judicial College of Victoria's Criminal Charge Book"
            },
            id=chunk["id"]
        )
        for chunk in chunks
    ]
    with open("documents/docs.pkl", "wb") as f:
        pickle.dump(documents, f)        

    vectorstore = Chroma(
        collection_name="Law_RAG",
        embedding_function=embedding_wrapper,
        persist_directory="./chroma_db"
    )

    for min_document_index in tqdm(range(0, len(documents), batch_size), desc="Processing document batches"):
        try:
            max_document_index = min(min_document_index + batch_size, total_chunks)
            vectorstore.add_documents(documents[min_document_index: max_document_index])
            successful_writes += max_document_index - min_document_index
        except Exception as e:
            failed_writes += max_document_index - min_document_index

    print(f"Finished processing chunks. Successful writes: {successful_writes}, Failed writes: {failed_writes}")        
    return successful_writes, failed_writes


def build_chroma_database(model_name="jinaai/jina-embeddings-v5-text-small"):
    legal_corpus_chunks = load_corpus()
    print("Loaded legal corpus chunks, total chunks:", len(legal_corpus_chunks))

    embedding_model = UniversalEmbeddingModel(model_name)
    embedding_wrapper = LocalEmbeddingWrapper(embedding_model)

    successful, failed = store_documents_in_batches(
        chunks=legal_corpus_chunks,
        embedding_wrapper=embedding_wrapper,
    )

    return successful, failed