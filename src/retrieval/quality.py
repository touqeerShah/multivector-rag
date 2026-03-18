from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


TOKEN_RE = re.compile(r"[a-z0-9]+")
MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "your",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def tokenize_terms(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def content_terms(text: str) -> List[str]:
    terms = [term for term in tokenize_terms(text) if term not in STOPWORDS]
    return [term for term in terms if len(term) > 1 or term.isdigit()]


def normalize_heading_text(text: str) -> str:
    text = MARKDOWN_HEADING_RE.sub("", normalize_whitespace(text))
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return normalize_whitespace(text)


def is_heading_only_chunk(text: str, heading: str = "") -> bool:
    compact = normalize_whitespace(text)
    if not compact:
        return True

    text_terms = tokenize_terms(compact)
    heading_norm = normalize_heading_text(heading)
    text_norm = normalize_heading_text(compact)

    if heading_norm and text_norm == heading_norm:
        return True

    if heading_norm and text_norm.startswith(heading_norm):
        remainder = text_norm[len(heading_norm) :].strip()
        if len(tokenize_terms(remainder)) <= 2:
            return True

    if (
        len(text_terms) <= 4
        and not re.search(r"[.!?;:]", compact)
        and not any(char.isdigit() for char in compact)
    ):
        return True

    return False


def is_low_value_chunk(text: str, heading: str = "") -> bool:
    compact = normalize_whitespace(text)
    if not compact:
        return True

    terms = tokenize_terms(compact)
    if not terms:
        return True

    if is_heading_only_chunk(compact, heading=heading):
        return True

    informative_terms = [term for term in terms if term not in STOPWORDS]
    has_digit = any(char.isdigit() for char in compact)

    if len(terms) < 3 and not has_digit:
        return True

    if len(informative_terms) <= 1 and len(terms) <= 6 and not has_digit:
        return True

    if len(set(terms)) == 1 and len(terms) <= 8:
        return True

    return False


def is_retrievable_row(row: Dict[str, Any]) -> bool:
    return not is_low_value_chunk(
        row.get("chunk_text", ""),
        heading=row.get("section_heading", ""),
    )


def filter_retrievable_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if is_retrievable_row(row)]


def lexical_overlap_count(query: str, text: str, heading: str = "") -> int:
    query_terms = set(content_terms(query))
    if not query_terms:
        query_terms = set(tokenize_terms(query))

    document_terms = set(content_terms(f"{heading} {text}"))
    if not document_terms:
        document_terms = set(tokenize_terms(f"{heading} {text}"))

    return len(query_terms & document_terms)
