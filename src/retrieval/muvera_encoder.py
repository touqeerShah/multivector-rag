from __future__ import annotations

from dataclasses import asdict
from typing import Any
import numpy as np
import sys
from pathlib import Path

# Make local muvera-py importable
MUVERA_PATH = Path("external/muvera-py").resolve()
if str(MUVERA_PATH) not in sys.path:
    sys.path.append(str(MUVERA_PATH))

from fde_generator import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde,
)


class MuveraEncoder:
    def __init__(
        self,
        dimension: int = 128,
        num_repetitions: int = 10,
        num_simhash_projections: int = 6,
        seed: int = 42,
    ):
        self.config = FixedDimensionalEncodingConfig(
            dimension=dimension,
            num_repetitions=num_repetitions,
            num_simhash_projections=num_simhash_projections,
            seed=seed,
        )

    def encode_query_multivectors(self, multi_vectors: np.ndarray) -> np.ndarray:
        return generate_query_fde(
            multi_vectors.astype(np.float32),
            self.config,
        )

    def encode_document_multivectors(self, multi_vectors: np.ndarray) -> np.ndarray:
        return generate_document_fde(
            multi_vectors.astype(np.float32),
            self.config,
        )

    def output_dim(self) -> int:
        # Generate once with a dummy shape to discover output size if needed
        dummy = np.zeros((1, self.config.dimension), dtype=np.float32)
        fde = generate_document_fde(dummy, self.config)
        return int(fde.shape[0])

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)