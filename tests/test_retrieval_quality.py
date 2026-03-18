from src.retrieval.quality import is_low_value_chunk, lexical_overlap_count


def test_is_low_value_chunk_filters_heading_only_content():
    assert is_low_value_chunk("Termination Clause", heading="Termination Clause") is True


def test_is_low_value_chunk_keeps_informative_content():
    assert (
        is_low_value_chunk(
            "The agreement may be terminated by either party with 30 days written notice.",
            heading="Termination",
        )
        is False
    )


def test_lexical_overlap_count_prefers_content_terms():
    overlap = lexical_overlap_count(
        query="termination notice period",
        text="The agreement requires 30 days notice before termination.",
        heading="Termination",
    )

    assert overlap >= 2
