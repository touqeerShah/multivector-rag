from typing import TypedDict, Any

class GraphState(TypedDict, total=False):
    query: str
    query_type: str
    bm25_hits: list[dict]
    dense_hits: list[dict]
    hybrid_hits: list[dict]
    muvera_hits: list[dict]
    text_reranked: list[dict]
    visual_reranked: list[dict]
    final_hits: list[dict]
    answer: str
    debug: dict[str, Any]