from __future__ import annotations

from dataclasses import dataclass
from typing import override

from ...types import FloatArray
from ..processor import UnsupervisedProcessor
from ..sample_scorer import SampleScorerLike


@dataclass(eq=False)
class SampleScoringProcessor(UnsupervisedProcessor):
    """
    Wraps estimators whose scores are not biased on their training samples.

    Attributes
    ----------
    estimator : SampleScorerLike
        The underlying estimator (structural type).
    """

    estimator: SampleScorerLike

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        _ = self.estimator.fit(X)
        return self._score_scaled(X)

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        return -self.estimator.score_samples(X)
