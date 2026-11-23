import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

OUT = Path("golden/golden.json")  # outside data/ so it is committed and the experiments stay reproducible
DISTRACTORS_OUT = Path("golden/distractors.json")
QUOTAS = {"bridge": 8, "comparison": 7}  # HotpotQA question types
N_DISTRACTOR_QUESTIONS = 300  # extra questions whose passages only pad the index, never tested on
MAX_SCAN = 20000  # stop scanning if quotas can't be filled
SEED = 42

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=2000)

picked, per_type = [], Counter()
distractors, seen_titles, n_distractor_questions = [], set(), 0
for scanned, r in enumerate(ds, 1):
    if scanned > MAX_SCAN:
        break
    titles = r["context"]["title"]

    if len(picked) == sum(QUOTAS.values()):  # test set is full: rows now only supply distractors
        if n_distractor_questions == N_DISTRACTOR_QUESTIONS:
            break
        n_distractor_questions += 1
        for i, (t, s) in enumerate(zip(titles, r["context"]["sentences"])):
            if t not in seen_titles:  # a repeated title could duplicate a gold passage that has no label
                seen_titles.add(t)
                distractors.append({"doc_id": f"d{r['id']}_p{i}", "title": t, "text": f"{t}: {' '.join(s).strip()}"})
        continue

    if per_type[r["type"]] >= QUOTAS.get(r["type"], 0):
        continue
    if r["answer"].strip().lower() in ("yes", "no"):  # a coin flip would score 50%, so it teaches nothing
        continue
    gold_titles = set(r["supporting_facts"]["title"])
    per_type[r["type"]] += 1
    seen_titles.update(titles)
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
DISTRACTORS_OUT.write_text(json.dumps(distractors, indent=2, ensure_ascii=False))
print(f"Saved {len(picked)} questions to {OUT} (scanned {scanned} rows)")
print(f"Saved {len(distractors)} distractor passages from {n_distractor_questions} extra questions to {DISTRACTORS_OUT}")
print("per type:", dict(per_type), "| levels:", dict(Counter(p["level"] for p in picked)))
for p in picked:
    print(f'- [{p["query_type"]}] {p["question"][:70]!r} -> {p["expected_output"]!r} | gold={len(p["gold_doc_ids"])}')
