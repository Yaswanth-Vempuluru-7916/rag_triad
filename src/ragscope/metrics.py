def recall_at_k(retrieved_docs : list[str], gold_docs : list[str], k : int) -> float :
    """
    Compute recall@k for a single query.
    retrieved_docs: list of doc_ids returned by the retriever, in order of decreasing relevance
    gold_docs: list of doc_ids that are relevant for the query
    k: number of top retrieved docs to consider

    retrieved = {"A", "B", "C"}
    gold = {"B", "C", "D"}
    print(retrieved & gold)

    Ans : {"B", "C"}

    """
    if not gold_docs:
        return 0.0
    retrieved_set = set(retrieved_docs[:k])
    gold_set = set(gold_docs)
    return len(retrieved_set & gold_set) / len(gold_set)


def precision_at_k(retrieved_docs : list[str], gold_docs : list[str], k : int) -> float :
    if not retrieved_docs or k <= 0:
        return 0.0

    gold_set = set(gold_docs)
    relevant_count = 0

    for doc_id in retrieved_docs[:k]:
        if doc_id in gold_set:
            relevant_count += 1

    return relevant_count / k

def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in gold:
            return 1 / rank
    return 0.0