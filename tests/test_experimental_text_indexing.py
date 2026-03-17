from __future__ import annotations

import time
from threading import Event

from src.services.experimental_text_indexing import ExperimentalTextIndexingService


def test_rebuild_colbert_index_updates_status_and_logs():
    service = ExperimentalTextIndexingService()

    service.store.all_text_rows = lambda: [{"id": "row-1"}, {"id": "row-2"}]
    service.exporter.export_collection_tsv = lambda rows: {
        "collection_tsv": "data/colbert/collection.tsv",
        "pid_mapping_json": "data/colbert/pid_mapping.json",
    }
    service.colbert.build_index = (
        lambda collection_tsv_path, overwrite, log_fn=None, num_rows=None: {
        "status": "indexed",
        "collection_tsv": collection_tsv_path,
        "index_name": "local_index",
        }
    )

    result = service.rebuild_colbert_index(overwrite=True)
    status = service.get_rebuild_status()

    assert result["rows_exported"] == 2
    assert status["state"] == "succeeded"
    assert status["result"]["index_name"] == "local_index"
    assert any(
        log["message"] == "Starting official ColBERT index build."
        for log in status["logs"]
    )


def test_start_rebuild_colbert_index_runs_in_background():
    service = ExperimentalTextIndexingService()
    started = Event()
    finish = Event()

    def fake_run(overwrite: bool):
        started.set()
        finish.wait(timeout=1)
        return {"status": "indexed", "index_name": "local_index"}

    service._run_rebuild_colbert_index = fake_run

    initial = service.start_rebuild_colbert_index(overwrite=True)
    assert initial["state"] == "running"

    assert started.wait(timeout=1)
    assert service.get_rebuild_status()["state"] == "running"

    finish.set()
    for _ in range(100):
        if service.get_rebuild_status()["state"] == "succeeded":
            break
        time.sleep(0.01)

    final = service.get_rebuild_status()
    assert final["state"] == "succeeded"
    assert final["result"]["index_name"] == "local_index"
