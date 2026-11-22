import json
from pathlib import Path
from statistics import mean

from ragscope.config import CHUNK_SIZE, OVERLAP, collection_name
from ragscope.embed import make_embedder
from ragscope.metrics import precision_at_k, recall_at_k, reciprocal_rank
from ragscope.retrieve import retrieve
from ragscope.store import get_collection

KS = (1, 3, 5)

golden = json.load(open("golden/golden.json"))
embedder = make_embedder()
name = collection_name(CHUNK_SIZE, OVERLAP, embedder.model_name)
col = get_collection("chroma_db", name)
if col.count() == 0:
    raise SystemExit("Index is empty. Run scripts/build_index.py first.")

rows = []
for q in golden:
    chunks = retrieve(col, embedder, q["question"], k=max(KS))
    ids = [c.doc_id for c in chunks]
    gold = q["gold_doc_ids"]
    row = {
        "id": q["id"],
        "question": q["question"],
        "expected_output": q["expected_output"],
        "gold_doc_ids": gold,
        "rr": reciprocal_rank(ids, set(gold)),
        "retrieved": [
            {
                "rank": rank,
                "doc_id": c.doc_id,
                "chunk_idx": c.chunk_idx,
                "similarity": round(c.similarity, 4),
                "is_gold": c.doc_id in gold,
                "text": c.text,
            }
            for rank, c in enumerate(chunks, 1)
        ],
    }
    for k in KS:
        row[f"recall@{k}"] = recall_at_k(ids, gold, k)
        row[f"precision@{k}"] = precision_at_k(ids, gold, k)
    rows.append(row)
    print(f'recall@5={row["recall@5"]:.2f}  rr={row["rr"]:.2f}  {q["question"][:60]}')

NON_METRICS = ("id", "question", "expected_output", "gold_doc_ids", "retrieved")
metrics = [m for m in rows[0] if m not in NON_METRICS]
averages = {m: mean(r[m] for r in rows) for m in metrics}
print(f"\nAverages over {len(rows)} questions ({name}):")
for m, v in averages.items():
    print(f"  {m:<14} {v:.3f}")

out = Path("results") / f"retriever_{name}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"collection": name, "averages": averages, "per_question": rows}, indent=2))
print("Saved", out)
