import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

OUT = Path("golden/golden.json")  # outside data/ so it is committed and the experiments stay reproducible
QUOTAS = {"bridge": 8, "comparison": 7}  # HotpotQA question types
MAX_SCAN = 20000  # stop scanning if quotas can't be filled
SEED = 42

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=2000)

picked, per_type = [], Counter()
for scanned, r in enumerate(ds, 1):
    if scanned > MAX_SCAN or len(picked) == sum(QUOTAS.values()):
        break
    if per_type[r["type"]] >= QUOTAS.get(r["type"], 0):
        continue
    if r["answer"].strip().lower() in ("yes", "no"):  # a coin flip would score 50%, so it teaches nothing
        continue
    titles = r["context"]["title"]
    gold_titles = set(r["supporting_facts"]["title"])
    per_type[r["type"]] += 1
    picked.append({
        "id": f"q{r['id']}",
        "question": r["question"].strip(),
        "expected_output": r["answer"].strip(),
        "query_type": r["type"],
        "level": r["level"],
        "gold_doc_ids": [f"q{r['id']}_p{i}" for i, t in enumerate(titles) if t in gold_titles],
        "passages": [
            {"doc_id": f"q{r['id']}_p{i}", "title": t, "text": f"{t}: {' '.join(s).strip()}"}
            for i, (t, s) in enumerate(zip(titles, r["context"]["sentences"]))
        ],
    })

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(picked, indent=2, ensure_ascii=False))
print(f"Saved {len(picked)} questions to {OUT} (scanned {scanned} rows)")
print("per type:", dict(per_type), "| levels:", dict(Counter(p["level"] for p in picked)))
for p in picked:
    print(f'- [{p["query_type"]}] {p["question"][:70]!r} -> {p["expected_output"]!r} | gold={len(p["gold_doc_ids"])}')
