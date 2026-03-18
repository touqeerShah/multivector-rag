from src.main_colbert import app as colbert_app
from src.main_colpali import app as colpali_app


def _paths(app):
    return {route.path for route in app.routes}


def test_main_colbert_exposes_text_and_experimental_routes():
    paths = _paths(colbert_app)

    assert "/health" in paths
    assert "/search" in paths
    assert "/answer" in paths
    assert "/experimental/muvera/reindex" in paths
    assert "/experimental/muvera/search" in paths
    assert "/experimental/muvera/real/reindex" in paths
    assert "/experimental/muvera/real/search" in paths
    assert "/experimental/search" in paths
    assert "/experimental/colbert/reindex" in paths
    assert "/experimental/colbert/reindex/background" in paths
    assert "/experimental/colbert/reindex/status" in paths
    assert "/visual/search" not in paths
    assert "/visual/embed-pages" not in paths


def test_main_colpali_exposes_text_and_visual_routes():
    paths = _paths(colpali_app)

    assert "/health" in paths
    assert "/search" in paths
    assert "/answer" in paths
    assert "/experimental/muvera/reindex" in paths
    assert "/experimental/muvera/search" in paths
    assert "/experimental/muvera/real/reindex" not in paths
    assert "/experimental/muvera/real/search" not in paths
    assert "/visual/search" in paths
    assert "/visual/embed-pages" in paths
    assert "/experimental/search" not in paths
    assert "/experimental/colbert/reindex" not in paths
