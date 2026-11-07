from __future__ import annotations

from dataclasses import dataclass
from typing import Final, override

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ...neighbors_metric import NeighborsMetric
from ...processor import AnomalyProcessor
from ...types import FloatArray, NumericArray

# Make the median and mean absolute deviations comparable to a normal standard deviation.
MEDIAN_ABSOLUTE_DEVIATION_SCALE: Final[float] = 1.4826
MEAN_ABSOLUTE_DEVIATION_SCALE: Final[float] = 1.2533


@dataclass(eq=False)
class MutualScoringProcessor(AnomalyProcessor):
    """
    Scores each sample against the other samples of the same batch.

    Anomalies are assumed to be rare and unlike the rest of the batch. Scores and
    thresholds are relative to the batch, so they are not comparable across batches.
    When over half of the scores are equal, the median absolute deviation is 0,
    so the mean absolute deviation measures the spread instead.

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
        if self.metric is NeighborsMetric.COSINE and not X.any(axis=1).all():
            raise ValueError("Vectors must not have zero length for cosine distance")
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
        spread: float = MEDIAN_ABSOLUTE_DEVIATION_SCALE * float(
            np.quantile(deviations, 0.5)
        )
        if spread == 0.0:
            spread = MEAN_ABSOLUTE_DEVIATION_SCALE * float(deviations.mean())
        return median + self.deviation_threshold * spread
