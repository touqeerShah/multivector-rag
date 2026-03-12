import re
from typing import List, Dict, Any


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def split_markdown_by_headings(markdown_text: str) -> List[Dict[str, Any]]:
    markdown_text = normalize_text(markdown_text)
    if not markdown_text:
        return []

    lines = markdown_text.splitlines()
    sections: List[Dict[str, Any]] = []

    current_heading = "Document Start"
    current_level = 0
    current_content: List[str] = []

    for line in lines:
        match = HEADING_RE.match(line.strip())
        if match:
            if current_content:
                sections.append(
                    {
                        "heading": current_heading,
                        "level": current_level,
                        "content": "\n".join(current_content).strip(),
                    }
                )
                current_content = []

            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
        else:
            current_content.append(line)

    if current_content:
        sections.append(
            {
                "heading": current_heading,
                "level": current_level,
                "content": "\n".join(current_content).strip(),
            }
        )

    return sections


def chunk_long_section(
    heading: str,
    content: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> List[str]:
    content = normalize_text(content)
    if not content:
        return []

    full_text = f"{heading}\n\n{content}".strip()

    if len(full_text) <= chunk_size:
        return [full_text]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def markdown_to_semantic_chunks(
    markdown_text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    sections = split_markdown_by_headings(markdown_text)
    final_chunks: List[Dict[str, Any]] = []

    for sec_idx, sec in enumerate(sections):
        sub_chunks = chunk_long_section(
            heading=sec["heading"],
            content=sec["content"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_idx, chunk in enumerate(sub_chunks):
            final_chunks.append(
                {
                    "section_heading": sec["heading"],
                    "section_level": sec["level"],
                    "section_index": sec_idx,
                    "chunk_index_in_section": chunk_idx,
                    "chunk_text": chunk,
                }
            )

    return final_chunks