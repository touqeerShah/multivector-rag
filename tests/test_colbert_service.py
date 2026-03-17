from unittest.mock import patch

import pytest

from src.retrieval.colbert_service import (
    ColBERTEnvironmentError,
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
    assert "uv run --project colbert-env uvicorn src.main:app --reload" in message
