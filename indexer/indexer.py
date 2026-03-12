import tqdm
from pathlib import Path
import gc

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

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
        # self.device = "cpu"

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


def store_documents_in_batches(
    collection,
    chunks,
    embedding_function,
    embed_batch_size=2,
    write_batch_size=20,
):
    total_chunks = len(chunks)
    successful_writes = 0
    failed_writes = []

    print(f"Starting to process {total_chunks} chunks")
    print(f"Embedding batch size: {embed_batch_size}")
    print(f"Write batch size: {write_batch_size}")
    print(f"Using custom embedding model: {embedding_function.model.model_name}")

    buffer_documents = []
    buffer_metadatas = []
    buffer_ids = []
    buffer_embeddings = []

    source = "Judicial College of Victoria's Criminal Charge Book"

    for i in tqdm(range(0, total_chunks, embed_batch_size), desc="Embedding batches"):
        batch_end = min(i + embed_batch_size, total_chunks)
        batch_chunks = chunks[i: batch_end]

        try:
            documents = batch_chunks["text"]
            metadatas = [
                {
                    "source": source,
                    "title": title,
                    "footnotes": footnotes if footnotes else "",
                }
                for title, footnotes in zip(batch_chunks["title"], batch_chunks["footnotes"])
            ]
            ids = batch_chunks["id"]

            embeddings = embedding_function(documents)

            buffer_documents.extend(documents)
            buffer_metadatas.extend(metadatas)
            buffer_ids.extend(ids)
            buffer_embeddings.extend(embeddings)

            # Write only when buffer is large enough
            if len(buffer_documents) >= write_batch_size:
                collection.add(
                    documents=buffer_documents,
                    metadatas=buffer_metadatas,
                    ids=buffer_ids,
                    embeddings=buffer_embeddings,
                )
                successful_writes += 1

                buffer_documents.clear()
                buffer_metadatas.clear()
                buffer_ids.clear()
                buffer_embeddings.clear()

                gc.collect()

        except Exception as e:
            print(f"Embedding batch {i // embed_batch_size + 1} failed: {e}")
            failed_writes.append((i, batch_end, str(e)))
            continue

    # Flush remaining buffered items
    if buffer_documents:
        try:
            collection.add(
                documents=buffer_documents,
                metadatas=buffer_metadatas,
                ids=buffer_ids,
                embeddings=buffer_embeddings,
            )
            successful_writes += 1
        except Exception as e:
            print(f"Final buffered write failed: {e}")
            failed_writes.append(("final", len(buffer_documents), str(e)))

    print(f"\nProcessing complete:")
    print(f"Successful writes: {successful_writes}")
    print(f"Failed writes: {len(failed_writes)}")

    return successful_writes, failed_writes


def build_chroma_database(model_name="jinaai/jina-embeddings-v5-text-small"):
    legal_corpus_chunks = load_corpus()
    print("Loaded legal corpus chunks, total chunks:", len(legal_corpus_chunks))

    embedding_model = UniversalEmbeddingModel(model_name)
    embedding_fn = ChromaEmbeddingFunction(embedding_model)
    print(
        f"Initialized embedding model: {model_name} on device: {embedding_model.device}"
    )

    client = chromadb.PersistentClient(path="./chroma_db")

    # Since you manually pass embeddings in collection.add(...),
    # you do not need embedding_function on the collection.
    collection = client.get_or_create_collection(name="Law_RAG")
    print("Collection loaded")

    successful, failed = store_documents_in_batches(
        collection=collection,
        chunks=legal_corpus_chunks,
        embedding_function=embedding_fn,
    )

    return successful, failed