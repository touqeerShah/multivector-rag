from src.ingest.chunking import chunk_text

def test_chunk_text():
    text = "hello " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)