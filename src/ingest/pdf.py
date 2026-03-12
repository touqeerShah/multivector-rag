from pathlib import Path
import fitz  # PyMuPDF

def extract_pdf_text_and_images(pdf_path: str, out_dir: str) -> list[dict]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_path = out_dir / f"{pdf_path.stem}_page_{i+1}.png"
        pix.save(str(image_path))

        pages.append({
            "page_number": i + 1,
            "text": text,
            "image_path": str(image_path),
            "source_pdf": str(pdf_path),
        })

    return pages