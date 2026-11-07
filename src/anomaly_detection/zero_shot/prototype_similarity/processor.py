from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from ...processor import AnomalyProcessor
from ...types import FloatArray, NumericArray
from ..unit_normalizer import UnitNormalizer


@dataclass(eq=False)
class PrototypeSimilarityProcessor(AnomalyProcessor):
    """
    Scores samples by their cosine similarity to reference embeddings.

    Attributes
    ----------
    normal_prototype : FloatArray
        The unit vector representing normal samples, with shape (n_features,).
    anomalous_prototype : FloatArray
        The unit vector representing anomalous samples, with shape (n_features,).
    temperature : float
        The softmax temperature applied to the similarities.
    decision_threshold : float
        The anomaly probability above which a sample is anomalous.
    """

    normal_prototype: FloatArray
    anomalous_prototype: FloatArray
    temperature: float
    decision_threshold: float

    @override
    def _compute_scores(self, X: NumericArray) -> FloatArray:
        n_features: int = self.normal_prototype.size
        if X.shape[1] != n_features:
            raise ValueError(
                f"X must have {n_features} features as the prototypes do, got {X.shape[1]}"
            )
        difference: FloatArray = self.anomalous_prototype - self.normal_prototype
        logits: FloatArray = UnitNormalizer.normalize(X) @ difference / self.temperature
        # Computed in log space because exp(-logits) overflows at low temperatures.
        return np.exp(-np.logaddexp(0.0, -logits))

    @override
    def _resolve_threshold(self, scores: FloatArray) -> float:
        return self.decision_threshold
