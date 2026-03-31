from tqdm import tqdm
from .ocr_loader import load_corpus
from ..qdrant_sparse import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, encode_sparse_text
from ..config import config
from ..model.embeddings import create_embedding_wrapper

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        Modifier,
        PointStruct,
        SparseVector,
        SparseVectorParams,
        VectorParams,
    )
except ImportError:
    QdrantClient = None
    Distance = None
    Modifier = None
    PointStruct = None
    SparseVector = None
    SparseVectorParams = None
    VectorParams = None


def store_documents_in_batches(documents, embedding_wrapper, storing_batch_size=config.storing_batch_size):
    if QdrantClient is None:
        raise ImportError(
            "qdrant-client is required to store documents in Qdrant. "
            "Install it with `pip install qdrant-client`."
        )

    total_chunks = len(documents)
    successful_writes = 0
    failed_writes = 0

    print(f"Starting to process {total_chunks} chunks")
    print(f"Batch size: {storing_batch_size}")
    print(f"Using custom embedding model: {embedding_wrapper.model.embedding_model_name}")

    client = QdrantClient(path=config.qdrant_db_link)

    if documents:
        sample_vector = embedding_wrapper.embed_query(documents[0].page_content)
        client.recreate_collection(
            collection_name=config.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=len(sample_vector),
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    modifier=Modifier.IDF,
                ),
            },
        )

    for min_document_index in tqdm(range(0, len(documents), storing_batch_size), desc="Processing document batches"):
        try:
            max_document_index = min(min_document_index + storing_batch_size, total_chunks)
            batch_documents = documents[min_document_index: max_document_index]
            batch_dense_vectors = embedding_wrapper.embed_documents([doc.page_content for doc in batch_documents])
            batch_points = [
                PointStruct(
                    id=doc.id or doc.metadata["chunk_id"],
                    vector={
                        DENSE_VECTOR_NAME: dense_vector,
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload={
                        "page_content": doc.page_content,
                        **doc.metadata,
                    },
                )
                for doc, dense_vector, (sparse_indices, sparse_values) in zip(
                    batch_documents,
                    batch_dense_vectors,
                    [encode_sparse_text(doc.page_content) for doc in batch_documents],
                )
            ]
            client.upsert(
                collection_name=config.collection_name,
                points=batch_points,
                wait=True,
            )
            successful_writes += max_document_index - min_document_index
        except Exception as e:
            failed_writes += max_document_index - min_document_index

    print(f"Finished processing chunks. Successful writes: {successful_writes}, Failed writes: {failed_writes}")        
    return successful_writes, failed_writes


def build_qdrant_database(embedding_model_name=config.default_embedding_model_name):
    corpus_documents = load_corpus()
    print("Loaded legal corpus chunks, total chunks:", len(corpus_documents))

    embedding_wrapper = create_embedding_wrapper(embedding_model_name)

    successful, failed = store_documents_in_batches(
        documents=corpus_documents,
        embedding_wrapper=embedding_wrapper,
    )

    return successful, failed
