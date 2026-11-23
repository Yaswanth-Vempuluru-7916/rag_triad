import json
from pathlib import Path

from ragscope.chunking import chunk
from ragscope.config import CHUNK_SIZE, OVERLAP, collection_name
from ragscope.embed import make_embedder
from ragscope.schemas import Document
from ragscope.store import add_chunks_to_chromadb, get_collection

golden = json.load(open("golden/golden.json"))
docs = [Document(p["doc_id"], q["id"], p["text"]) for q in golden for p in q["passages"]]
distractors = Path("golden/distractors.json")  # extra passages that make retrieval harder
if distractors.exists():
    docs += [Document(p["doc_id"], "distractor", p["text"]) for p in json.load(open(distractors))]
chunks = [c for d in docs for c in chunk(d, CHUNK_SIZE, OVERLAP)]

embedder = make_embedder()
name = collection_name(CHUNK_SIZE, OVERLAP, embedder.model_name)
col = get_collection("chroma_db", name)
added = add_chunks_to_chromadb(col, chunks, embedder)
print(f"{name}: {added} new chunks embedded, {len(chunks) - added} already stored, {col.count()} total")
