from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

from src.ingest.pdf import extract_pdf_markdown_and_images, extract_txt_text
from src.ingest.chunking import chunk_text, markdown_to_semantic_chunks
# from src.retrieval.dense import DenseEmbedder
from src.retrieval.store import RetrievalStore
from src.retrieval.bm25 import BM25Index


class IndexingService:
    def __init__(self, embedder):
        self.embedder = embedder
        self.store = RetrievalStore()
        self.bm25 = BM25Index()

    def index_file(self, file_path: str) -> Dict[str, Any]:
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()

        if suffix == ".pdf":
            pages = extract_pdf_markdown_and_images(
                pdf_path=str(file_path_obj),
                output_dir="data/processed",
            )
            rows = self._build_rows_from_pdf(
                doc_id=file_path_obj.stem,
                source_file=str(file_path_obj),
                pages=pages,
            )
        elif suffix in {".txt", ".md"}:
            text = extract_txt_text(str(file_path_obj))
            rows = self._build_rows_from_text(
                doc_id=file_path_obj.stem,
                source_file=str(file_path_obj),
                text=text,
            )
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        self.store.add_text_rows(rows)
        self.rebuild_bm25()

        return {
            "status": "indexed",
            "file": str(file_path_obj),
            "chunks_indexed": len(rows),
        }

    def rebuild_bm25(self) -> None:
        rows = self.store.all_text_rows()
        self.bm25.build(rows)

    def _build_rows_from_text(
        self, doc_id: str, source_file: str, text: str
    ) -> List[Dict[str, Any]]:
        chunks = chunk_text(text)
        vectors = self.embedder.embed_texts(chunks) if chunks else []

        rows: List[Dict[str, Any]] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            rows.append(
                {
                    "id": f"{doc_id}-chunk-{idx}-{uuid4().hex[:8]}",
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "file_type": "text",
                    "page_number": 1,
                    "chunk_index": idx,
                    "section_heading": "",
                    "section_level": 0,
                    "chunk_text": chunk,
                    "image_path": "",
                    "vector": vector,
                }
            )
        return rows

    def _build_rows_from_pdf(
        self,
        doc_id: str,
        source_file: str,
        pages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for page in pages:
            semantic_chunks = markdown_to_semantic_chunks(page["markdown"])

            if semantic_chunks:
                texts = [c["chunk_text"] for c in semantic_chunks]
                vectors = self.embedder.embed_texts(texts) if texts else []

                for idx, (chunk_info, vector) in enumerate(
                    zip(semantic_chunks, vectors)
                ):
                    rows.append(
                        {
                            "id": f"{doc_id}-p{page['page_number']}-s{chunk_info['section_index']}-c{idx}-{uuid4().hex[:8]}",
                            "doc_id": doc_id,
                            "source_file": source_file,
                            "file_type": "pdf",
                            "page_number": page["page_number"],
                            "chunk_index": idx,
                            "section_heading": chunk_info["section_heading"],
                            "section_level": chunk_info["section_level"],
                            "chunk_text": chunk_info["chunk_text"],
                            "image_path": page["image_path"],
                            "vector": vector,
                        }
                    )
                continue

            # Fallback if markdown structure is weak or empty
            fallback_chunks = chunk_text(page["text"])
            vectors = (
                self.embedder.embed_texts(fallback_chunks) if fallback_chunks else []
            )

            for idx, (chunk, vector) in enumerate(zip(fallback_chunks, vectors)):
                rows.append(
                    {
                        "id": f"{doc_id}-p{page['page_number']}-fallback-{idx}-{uuid4().hex[:8]}",
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "file_type": "pdf",
                        "page_number": page["page_number"],
                        "chunk_index": idx,
                        "section_heading": "",
                        "section_level": 0,
                        "chunk_text": chunk,
                        "image_path": page["image_path"],
                        "vector": vector,
                    }
                )

        return rows
