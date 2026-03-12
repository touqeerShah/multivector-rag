def reciprocal_rank_fusion(*rank_lists, k: int = 60):
    scores = {}
    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return [
        {"id": doc_id, "score": score}
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
