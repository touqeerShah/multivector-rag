VISUAL_HINTS = {
    "table", "figure", "chart", "diagram", "layout", "page",
    "form", "invoice", "scan", "screenshot", "image"
}

def classify_query(query: str) -> str:
    q = query.lower()
    if any(word in q for word in VISUAL_HINTS):
        return "visual"
    return "text"

def choose_rerankers(query: str) -> list[str]:
    query_type = classify_query(query)
    if query_type == "visual":
        return ["colqwen", "colbert"]
    return ["colbert"]