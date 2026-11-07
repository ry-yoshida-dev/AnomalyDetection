from __future__ import annotations

from dataclasses import dataclass
from typing import Final, override

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ...neighbors_metric import NeighborsMetric
from ...processor import AnomalyProcessor
from ...types import FloatArray, NumericArray

# Makes the median absolute deviation comparable to a normal standard deviation.
MEDIAN_ABSOLUTE_DEVIATION_SCALE: Final[float] = 1.4826


@dataclass(eq=False)
class MutualScoringProcessor(AnomalyProcessor):
    """
    Scores each sample against the other samples of the same batch.

    Anomalies are assumed to be rare and unlike the rest of the batch. Scores and
    thresholds are relative to the batch, so they are not comparable across batches.

    Attributes
    ----------
    n_neighbors : int
        Number of other samples averaged.
    metric : NeighborsMetric
        The distance metric.
    deviation_threshold : float
        Scaled median absolute deviations above the median score beyond which a
        sample is anomalous.
    n_jobs : int | None
        Number of parallel jobs.
    """

    n_neighbors: int
    metric: NeighborsMetric
    deviation_threshold: float
    n_jobs: int | None

    @override
    def _compute_scores(self, X: NumericArray) -> FloatArray:
        if X.shape[0] <= self.n_neighbors:
            raise ValueError(
                f"X must have more than {self.n_neighbors} samples, got {X.shape[0]}"
            )
        index: NearestNeighbors = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric.value,
            n_jobs=self.n_jobs,
        ).fit(X)
        distances: FloatArray = index.kneighbors()[0]
        return distances.mean(axis=1)

    @override
    def _resolve_threshold(self, scores: FloatArray) -> float:
        median: float = float(np.quantile(scores, 0.5))
        deviations: FloatArray = np.abs(scores - median)
        deviation: float = float(np.quantile(deviations, 0.5))
        return (
            median
            + self.deviation_threshold * MEDIAN_ABSOLUTE_DEVIATION_SCALE * deviation
        )
