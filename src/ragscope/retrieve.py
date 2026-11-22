from ragscope.embed import Embedder
from ragscope.schemas import RetrievedChunk


def retrieve(collection, embedder: Embedder, query: str, k: int) -> list[RetrievedChunk]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be > 0")

    k = min(k, collection.count())  # asking for more than exists would error
    if k == 0:
        return []

    res = collection.query(query_embeddings=embedder.embed([query]).tolist(), n_results=k)
    return [
        RetrievedChunk(m["doc_id"], m["chunk_idx"], text, similarity=1 - dist)
        for m, text, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0])
    ]
