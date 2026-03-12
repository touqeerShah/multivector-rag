from typing import List, Dict, Any


def reciprocal_rank_fusion(
    bm25_hits: List[Dict[str, Any]],
    dense_hits: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    scores: dict[str, float] = {}
    merged_docs: dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(bm25_hits, start=1):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        merged_docs[doc_id] = {**item}

    for rank, item in enumerate(dense_hits, start=1):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id in merged_docs:
            merged_docs[doc_id].update(item)
        else:
            merged_docs[doc_id] = {**item}

    final = []
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = merged_docs[doc_id]
        item["rrf_score"] = score
        final.append(item)

    return final