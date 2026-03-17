from unittest.mock import patch

import pytest

from src.retrieval.colbert_service import (
    ColBERTEnvironmentError,
    OfficialColBERTService,
    ensure_colbert_runtime_compatible,
)


def test_colbert_runtime_allows_transformers_4_stack():
    versions = {
        "transformers": "4.48.3",
        "sentence-transformers": "3.4.1",
    }

    with patch("src.retrieval.colbert_service._installed_version", side_effect=versions.get):
        ensure_colbert_runtime_compatible()


def test_colbert_runtime_rejects_transformers_5_stack():
    versions = {
        "transformers": "5.3.0",
        "sentence-transformers": "5.3.0",
    }

    with patch("src.retrieval.colbert_service._installed_version", side_effect=versions.get):
        with pytest.raises(ColBERTEnvironmentError) as exc_info:
            ensure_colbert_runtime_compatible()

    message = str(exc_info.value)
    assert "transformers=5.3.0" in message
    assert "dedicated ColBERT project environment" in message


def test_safe_partition_count_caps_small_corpora():
    safe = OfficialColBERTService._safe_partition_count(
        requested=32,
        sample_embeddings=9,
        max_partitions=128,
    )

    assert safe == 8


def test_choose_partitions_prefers_tiny_values_for_small_corpora():
    chosen = OfficialColBERTService._choose_partitions(
        num_rows=3,
        est_embeddings=9,
        max_partitions=128,
    )

    assert chosen == 1


def test_choose_partitions_scales_gradually_for_medium_corpora():
    chosen = OfficialColBERTService._choose_partitions(
        num_rows=400,
        est_embeddings=6000,
        max_partitions=128,
    )

    assert chosen == 8
