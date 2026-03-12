from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pymupdf4llm


def extract_pdf_markdown_and_images(
    pdf_path: str, output_dir: str
) -> List[Dict[str, Any]]:
    pdf_file = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract markdown per page
    md_pages = pymupdf4llm.to_markdown(
        str(pdf_file),
        page_chunks=True,
        write_images=False,
        embed_images=False,
        show_progress=False,
    )

    # Render page images
    doc = fitz.open(pdf_file)

    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(doc):
        page_number = i + 1

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_path = out_dir / f"{pdf_file.stem}_page_{page_number}.png"
        pix.save(str(image_path))

        md_info = md_pages[i] if i < len(md_pages) else {}
        markdown_text = (
            md_info.get("text")
            or md_info.get("md")
            or md_info.get("markdown")
            or ""
        )

        pages.append(
            {
                "page_number": page_number,
                "markdown": markdown_text,
                "text": page.get_text("text") or "",
                "image_path": str(image_path),
                "source_file": str(pdf_file),
            }
        )

    return pages


def extract_txt_text(txt_path: str) -> str:
    return Path(txt_path).read_text(encoding="utf-8", errors="ignore")