
import argparse
import csv
from pathlib import Path
from datetime import datetime

import chromadb

parser = argparse.ArgumentParser(description="Dump a Chroma collection to terminal and CSV")
parser.add_argument("--path", default="chroma_db")
parser.add_argument("--name", required=True)
parser.add_argument("--out", default="data/store_dump.csv")
args = parser.parse_args()

col = chromadb.PersistentClient(path=args.path).get_collection(args.name)
data = col.get(include=["documents", "metadatas"])

rows = [
    {"id": i, "doc_id": m["doc_id"], "chunk_idx": m["chunk_idx"], "text": t}
    for i, m, t in zip(data["ids"], data["metadatas"], data["documents"])
]

print(f"{col.name}: {col.count()} chunks")

for r in rows[:10]:
    print(f'{r["id"]:<12} {r["text"][:80]!r}')

# Generate timestamped filename
out_path = Path(args.out)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

filename = f"{out_path.stem}_{timestamp}{out_path.suffix}"
final_path = out_path.parent / filename

final_path.parent.mkdir(parents=True, exist_ok=True)

with open(final_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "doc_id", "chunk_idx", "text"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Full dump written to {final_path}")
