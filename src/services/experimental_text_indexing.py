from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, Any
from uuid import uuid4

from src.retrieval.store import RetrievalStore
from src.retrieval.collection_export import CollectionExporter
from src.retrieval.colbert_service import OfficialColBERTService


class ExperimentalTextIndexingService:
    def __init__(self):
        self.store = RetrievalStore()
        self.exporter = CollectionExporter()
        self.colbert = OfficialColBERTService()
        self._status_lock = Lock()
        self._rebuild_status = self._empty_status()

    def rebuild_colbert_index(self, overwrite: bool = True) -> Dict[str, Any]:
        self._mark_running(overwrite=overwrite)
        try:
            result = self._run_rebuild_colbert_index(overwrite=overwrite)
        except Exception as exc:
            self._mark_failed(exc)
            raise

        self._mark_succeeded(result)
        return result

    def start_rebuild_colbert_index(self, overwrite: bool = True) -> Dict[str, Any]:
        with self._status_lock:
            if self._rebuild_status["state"] == "running":
                return self._snapshot_status()

            self._rebuild_status = self._new_running_status(overwrite=overwrite)

        worker = Thread(
            target=self._run_rebuild_in_background,
            kwargs={"overwrite": overwrite},
            daemon=True,
            name="colbert-reindex",
        )
        worker.start()

        self._append_log("Background ColBERT reindex worker started.")
        return self._snapshot_status()

    def get_rebuild_status(self) -> Dict[str, Any]:
        return self._snapshot_status()

    def _run_rebuild_in_background(self, overwrite: bool) -> None:
        try:
            result = self._run_rebuild_colbert_index(overwrite=overwrite)
        except Exception as exc:
            self._mark_failed(exc)
            return

        self._mark_succeeded(result)

    def _run_rebuild_colbert_index(self, overwrite: bool) -> Dict[str, Any]:
        self._append_log("Loading text rows from the retrieval store.")
        rows = self.store.all_text_rows()
        self._append_log(f"Loaded {len(rows)} text rows.")

        self._append_log("Exporting collection TSV and PID mapping for ColBERT.")
        export_result = self.exporter.export_collection_tsv(rows)
        self._append_log(
            f"Collection export complete: {export_result['collection_tsv']}."
        )

        # Resolve the index output directory so the watcher can poll it.
        index_dir = Path("experiments") / "local" / "indexes" / "local_index"

        done_event = Event()
        watcher = Thread(
            target=self._watch_index_progress,
            args=(index_dir, done_event),
            daemon=True,
            name="colbert-index-watcher",
        )
        watcher.start()

        self._append_log("Starting official ColBERT index build.")
        try:
            result = self.colbert.build_index(
                collection_tsv_path=export_result["collection_tsv"],
                overwrite=overwrite,
                log_fn=self._append_log,
                num_rows=len(rows),
            )
        finally:
            done_event.set()   # always stop the watcher
            watcher.join(timeout=5)

        self._append_log("Official ColBERT index build finished.")

        result["rows_exported"] = len(rows)
        result["pid_mapping_json"] = export_result["pid_mapping_json"]
        return result

    # ------------------------------------------------------------------
    # Filesystem watcher — detects ColBERT phase completion via output files
    # ------------------------------------------------------------------
    #
    # ColBERT's worker subprocesses write specific files at the end of each
    # phase.  Polling for these is the only reliable way to get phase-level
    # visibility when build_index() runs in a subprocess (monkey-patches on
    # CollectionIndexer methods don't cross the fork/spawn boundary).
    #
    # Phase milestones:
    #   plan.json       → setup complete    (num_partitions decided)
    #   centroids.pt    → train complete    (K-means finished)
    #   0.codes.pt      → index complete    (passages encoded)
    #   metadata.json   → finalize complete (IVF built, index ready)
    # ------------------------------------------------------------------
    _COLBERT_MILESTONES = [
        ("plan.json",    "[Phase 1/4] Setup complete: index plan written (num_partitions decided)."),
        ("centroids.pt", "[Phase 2/4] Train complete: K-means centroids computed and saved."),
        ("0.codes.pt",   "[Phase 3/4] Index complete: passage embeddings encoded and saved."),
        ("metadata.json","[Phase 4/4] Finalize complete: IVF built, index is ready."),
    ]

    def _watch_index_progress(self, index_dir: Path, done: Event) -> None:
        seen: set[str] = set()
        while not done.is_set():
            if index_dir.exists():
                for filename, message in self._COLBERT_MILESTONES:
                    if filename not in seen and (index_dir / filename).exists():
                        seen.add(filename)
                        self._append_log(message)
            done.wait(timeout=2)  # poll every 2 s; exits immediately when done is set

    def _mark_running(self, overwrite: bool) -> None:
        with self._status_lock:
            self._rebuild_status = self._new_running_status(overwrite=overwrite)

    def _mark_succeeded(self, result: Dict[str, Any]) -> None:
        with self._status_lock:
            self._rebuild_status["state"] = "succeeded"
            self._rebuild_status["finished_at"] = self._now()
            self._rebuild_status["result"] = result
            self._rebuild_status["error"] = None
            self._append_log_unlocked("ColBERT reindex completed successfully.")

    def _mark_failed(self, exc: Exception) -> None:
        with self._status_lock:
            self._rebuild_status["state"] = "failed"
            self._rebuild_status["finished_at"] = self._now()
            self._rebuild_status["error"] = str(exc)
            self._rebuild_status["result"] = None
            self._append_log_unlocked(f"ColBERT reindex failed: {exc}")

    def _append_log(self, message: str) -> None:
        with self._status_lock:
            self._append_log_unlocked(message)

    def _append_log_unlocked(self, message: str) -> None:
        self._rebuild_status["logs"].append(
            {
                "timestamp": self._now(),
                "message": message,
            }
        )
        self._rebuild_status["logs"] = self._rebuild_status["logs"][-50:]

    def _snapshot_status(self) -> Dict[str, Any]:
        with self._status_lock:
            return {
                **self._rebuild_status,
                "logs": list(self._rebuild_status["logs"]),
                "result": (
                    dict(self._rebuild_status["result"])
                    if isinstance(self._rebuild_status["result"], dict)
                    else self._rebuild_status["result"]
                ),
            }

    def _new_running_status(self, overwrite: bool) -> Dict[str, Any]:
        return {
            "job_id": uuid4().hex,
            "state": "running",
            "started_at": self._now(),
            "finished_at": None,
            "overwrite": overwrite,
            "result": None,
            "error": None,
            "logs": [
                {
                    "timestamp": self._now(),
                    "message": "ColBERT reindex requested.",
                }
            ],
        }

    def _empty_status(self) -> Dict[str, Any]:
        return {
            "job_id": None,
            "state": "idle",
            "started_at": None,
            "finished_at": None,
            "overwrite": None,
            "result": None,
            "error": None,
            "logs": [],
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
