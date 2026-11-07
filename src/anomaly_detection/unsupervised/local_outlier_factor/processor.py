from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.neighbors import LocalOutlierFactor

from ...types import FloatArray
from ..processor import UnsupervisedProcessor


@dataclass(eq=False)
class LocalOutlierFactorProcessor(UnsupervisedProcessor):
    """
    Scores samples by their local density relative to their neighbors.

    Attributes
    ----------
    estimator : LocalOutlierFactor
        The underlying estimator, in novelty mode.
    """

    estimator: LocalOutlierFactor

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        _ = self.estimator.fit(X)
        return -self.estimator.negative_outlier_factor_

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        return -self.estimator.score_samples(X)
