from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.covariance import LedoitWolf

from ...types import FloatArray
from ..processor import UnsupervisedProcessor


@dataclass(eq=False)
class MahalanobisProcessor(UnsupervisedProcessor):
    """
    Scores samples by their squared Mahalanobis distance from the training mean.

    Attributes
    ----------
    estimator : LedoitWolf
        The underlying shrinkage covariance estimator.
    """

    estimator: LedoitWolf

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        _ = self.estimator.fit(X)
        return self._score_scaled(X)

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        return self.estimator.mahalanobis(X)
