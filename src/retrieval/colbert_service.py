from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
from typing import List, Dict, Any
import json

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher


class ColBERTEnvironmentError(RuntimeError):
    pass


def _major_version(version: str | None) -> int | None:
    if not version:
        return None

    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _installed_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def ensure_colbert_runtime_compatible() -> None:
    transformers_version = _installed_version("transformers")
    sentence_transformers_version = _installed_version("sentence-transformers")

    incompatible = []
    if _major_version(transformers_version) and _major_version(transformers_version) >= 5:
        incompatible.append(f"transformers={transformers_version}")
    if _major_version(sentence_transformers_version) and _major_version(sentence_transformers_version) >= 4:
        incompatible.append(f"sentence-transformers={sentence_transformers_version}")

    if not incompatible:
        return

    versions = ", ".join(filter(None, incompatible))
    raise ColBERTEnvironmentError(
        "Official ColBERT indexing/search must run in the dedicated ColBERT environment. "
        f"Detected incompatible packages: {versions}. "
        "Use the dedicated ColBERT project environment."
    )


class OfficialColBERTService:
    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
        experiment_root: str = "experiments",
        experiment_name: str = "local",
        index_name: str = "local_index",
        mapping_path: str = "data/colbert/pid_mapping.json",
        max_partitions: int = 128,
    ):
        self.checkpoint = checkpoint
        self.experiment_root = experiment_root
        self.experiment_name = experiment_name
        self.index_name = index_name
        self.mapping_path = mapping_path
        self.max_partitions = max_partitions

    @staticmethod
    def _safe_partition_count(
        requested: int,
        sample_embeddings: int,
        max_partitions: int,
    ) -> int:
        capped = max(1, min(requested, sample_embeddings, max_partitions))

        power_of_two = 1
        while (power_of_two * 2) <= capped:
            power_of_two *= 2

        return power_of_two

    def build_index(
        self,
        collection_tsv_path: str,
        overwrite: bool = False,
        log_fn=None,
    ) -> Dict[str, Any]:
        """Build a ColBERT index, calling log_fn(msg) at each major phase."""
        ensure_colbert_runtime_compatible()
        Path(self.experiment_root).mkdir(parents=True, exist_ok=True)

        # Force HuggingFace to use the local cache without scanning remote/index files.
        # Without this, HF walks the entire ~/.cache/huggingface/hub directory
        # (can be 34GB+ with many models) doing thousands of lstat() calls before
        # even starting to load the model — causing a 10-30 minute hang.
        _prev_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        _prev_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        def _log(msg: str) -> None:
            if log_fn is not None:
                try:
                    log_fn(msg)
                except Exception:
                    pass

        from colbert.indexing.collection_indexer import CollectionIndexer

        original_setup = CollectionIndexer.setup
        original_train = CollectionIndexer.train
        original_index = CollectionIndexer.index
        original_finalize = CollectionIndexer.finalize

        def patched_setup(indexer_self):
            _log("[Phase 1/4] Setup: sampling passages and estimating corpus size…")
            original_setup(indexer_self)

            sample_embeddings = getattr(indexer_self, "num_sample_embs", None)
            if hasattr(sample_embeddings, "item"):
                sample_embeddings = int(sample_embeddings.item())

            if isinstance(sample_embeddings, int) and sample_embeddings > 0:
                requested = getattr(indexer_self, "num_partitions", 1)
                safe_partitions = self._safe_partition_count(
                    requested=requested,
                    sample_embeddings=sample_embeddings,
                    max_partitions=self.max_partitions,
                )

                if safe_partitions != requested:
                    indexer_self.num_partitions = safe_partitions
                    indexer_self._save_plan()
                    _log(
                        "[Phase 1/4] Setup adjusted for small corpus: "
                        f"sample_embeddings={sample_embeddings}, "
                        f"requested_partitions={requested}, "
                        f"using_partitions={safe_partitions}."
                    )

            parts = getattr(indexer_self, "num_partitions", "?")
            embs = getattr(indexer_self, "num_embeddings_est", "?")
            _log(
                f"[Phase 1/4] Setup complete: num_partitions={parts}, "
                f"estimated_embeddings={int(embs) if embs != '?' else '?'}."
            )

        def patched_train(indexer_self, shared_lists):
            parts = getattr(indexer_self, "num_partitions", "?")
            _log(f"[Phase 2/4] Train: running K-means clustering ({parts} centroids)…")
            original_train(indexer_self, shared_lists)
            _log("[Phase 2/4] Train complete: centroids computed and codec saved.")

        def patched_index(indexer_self):
            _log("[Phase 3/4] Index: encoding all passages into residual embeddings…")
            original_index(indexer_self)
            _log("[Phase 3/4] Index complete: all chunks encoded and saved.")

        def patched_finalize(indexer_self):
            _log("[Phase 4/4] Finalize: building IVF and writing metadata…")
            original_finalize(indexer_self)
            _log("[Phase 4/4] Finalize complete: index is ready.")

        CollectionIndexer.setup = patched_setup
        CollectionIndexer.train = patched_train
        CollectionIndexer.index = patched_index
        CollectionIndexer.finalize = patched_finalize

        try:
            with Run().context(
                RunConfig(
                    nranks=1,
                    experiment=self.experiment_name,
                    root=self.experiment_root,
                )
            ):
                config = ColBERTConfig(
                    root=self.experiment_root,
                    nbits=2,
                )
                indexer = Indexer(
                    checkpoint=self.checkpoint,
                    config=config,
                )
                _log("Checkpoint loaded. Starting ColBERT indexing pipeline…")
                indexer.index(
                    name=self.index_name,
                    collection=collection_tsv_path,
                    overwrite=overwrite,
                )
                _log("" \
                    "ColBERT indexing pipeline complete. Finalizing index and writing metadata…")

        except Exception as exc:
            _log(f"[ERROR] ColBERT index build failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            # Restore HuggingFace offline env vars
            if _prev_tf_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf_offline
            if _prev_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = _prev_hf_offline
            # Restore original ColBERT phase methods
            CollectionIndexer.setup = original_setup
            CollectionIndexer.train = original_train
            CollectionIndexer.index = original_index
            CollectionIndexer.finalize = original_finalize

        return {
            "status": "indexed",
            "collection_tsv": collection_tsv_path,
            "index_name": self.index_name,
            "max_partitions": self.max_partitions,
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        ensure_colbert_runtime_compatible()
        pid_map = self._load_pid_map()

        with Run().context(
            RunConfig(
                nranks=1,
                experiment=self.experiment_name,
                root=self.experiment_root,
            )
        ):
            config = ColBERTConfig(root=self.experiment_root)
            searcher = Searcher(index=self.index_name, config=config)
            pids, ranks, scores = searcher.search(query, k=top_k)

        results = []
        for pid, rank, score in zip(pids, ranks, scores):
            pid_str = str(pid)
            real_id = pid_map.get(pid_str, pid_str)

            results.append(
                {
                    "pid": pid_str,
                    "id": real_id,
                    "rank": int(rank),
                    "colbert_score": float(score),
                }
            )
        return results

    def _load_pid_map(self) -> Dict[str, str]:
        path = Path(self.mapping_path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
