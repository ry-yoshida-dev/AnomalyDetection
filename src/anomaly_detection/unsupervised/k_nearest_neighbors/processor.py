from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.neighbors import NearestNeighbors

from ...types import FloatArray
from ..processor import UnsupervisedProcessor


@dataclass(eq=False)
class KNearestNeighborsProcessor(UnsupervisedProcessor):
    """
    Scores samples by their mean distance to the nearest training samples.

    Attributes
    ----------
    estimator : NearestNeighbors
        The underlying neighbor index.
    """

    estimator: NearestNeighbors

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        _ = self.estimator.fit(X)
        # Querying without X excludes each sample from its own neighbors;
        # otherwise the threshold would be too low for unseen samples.
        distances: FloatArray = self.estimator.kneighbors()[0]
        return distances.mean(axis=1)

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        distances: FloatArray = self.estimator.kneighbors(X)[0]
        return distances.mean(axis=1)
