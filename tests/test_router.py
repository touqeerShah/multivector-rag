from src.rerank.router import classify_query

def test_visual_query():
    assert classify_query("find the pricing table on page 2") == "visual"

def test_text_query():
    assert classify_query("what is the termination clause") == "text"