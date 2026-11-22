import chromadb
from ragscope.embed import Embedder
from ragscope.schemas import Chunk


def get_collection(path: str, name: str):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def add_chunks(collection, chunks: list[Chunk], embedder: Embedder, batch_size: int = 256):
    chunks = [c for c in chunks if c.text.strip()]
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[f"{c.doc_id}:{c.chunk_idx}" for c in batch],
            documents=[c.text for c in batch],
            embeddings=embedder.embed([c.text for c in batch]).tolist(),
            metadatas=[{"doc_id": c.doc_id, "chunk_idx": c.chunk_idx} for c in batch],
        )
