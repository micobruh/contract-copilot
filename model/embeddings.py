import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from utils.utils import determine_device, determine_dtype, determine_model_path


class UniversalEmbeddingModel:
    def __init__(self, embedding_model_name, embedding_models_root="embedding_models"):
        self.embedding_model_name = embedding_model_name

        self.device = determine_device()
        self.dtype = determine_dtype(self.device)

        local_embedding_model_map = {
            "BAAI/bge-m3": "bge-m3",
            "Qwen/Qwen3-Embedding-4B": "Qwen3-Embedding-4B",
            "jinaai/jina-embeddings-v5-text-small": "jina-embeddings-v5-text-small",
        }
        
        self.embedding_model_path_str = determine_model_path(
            self.embedding_model_name,
            local_embedding_model_map,
            embedding_models_root,
        )

        if self.embedding_model_name == "BAAI/bge-m3":
            self.backend = "sentence_transformer"
            self.model = SentenceTransformer(
                self.embedding_model_path_str,
                device=self.device,
                local_files_only=True,
            )

        elif self.embedding_model_name == "Qwen/Qwen3-Embedding-4B":
            self.backend = "sentence_transformer"
            self.model = SentenceTransformer(
                self.embedding_model_path_str,
                device=self.device,
                local_files_only=True,
                model_kwargs={"dtype": self.dtype},
                tokenizer_kwargs={"padding_side": "left"},
            )

        elif self.embedding_model_name == "jinaai/jina-embeddings-v5-text-small":
            self.backend = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_path_str,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                self.embedding_model_path_str,
                trust_remote_code=True,
                dtype=self.dtype,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if self.backend == "sentence_transformer":
            if self.embedding_model_name == "Qwen/Qwen3-Embedding-4B":
                return self.model.encode(texts, prompt_name="query")
            return self.model.encode(texts)

        elif self.backend == "transformers":
            with torch.no_grad():
                return self.model.encode(
                    texts=texts,
                    task="retrieval",
                    prompt_name="document",
                )
    

class LocalEmbeddingWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    

def create_embedding_wrapper(embedding_model_name="BAAI/bge-m3"):
    embedding_model = UniversalEmbeddingModel(embedding_model_name)
    embedding_wrapper = LocalEmbeddingWrapper(embedding_model)
    return embedding_wrapper