import chromadb
from ragscope.embed import Embedder
from ragscope.schemas import Chunk


def get_collection(path: str, name: str):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def add_chunks_to_chromadb(collection, chunks: list[Chunk], embedder: Embedder, batch_size: int = 256) -> int:
    """Embed and store only chunks whose id is not stored yet. Returns how many were added."""
    chunks = [c for c in chunks if c.text.strip()]
    added = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        ids = [f"{c.doc_id}:{c.chunk_idx}" for c in batch]
        stored = set(collection.get(ids=ids, include=[])["ids"])
        new = [(id_, c) for id_, c in zip(ids, batch) if id_ not in stored]
        if not new:
            continue
        collection.upsert(
            ids=[id_ for id_, _ in new],
            documents=[c.text for _, c in new],
            embeddings=embedder.embed([c.text for _, c in new]).tolist(),
            metadatas=[{"doc_id": c.doc_id, "chunk_idx": c.chunk_idx} for _, c in new],
        )
        added += len(new)
    return added
