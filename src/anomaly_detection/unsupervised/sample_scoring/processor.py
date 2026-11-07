from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from sklearn.base import clone

from ...types import FloatArray
from ..processor import UnsupervisedProcessor
from ..sample_scorer import SampleScorerLike
from ..training_folds import TrainingFolds


@dataclass(eq=False)
class SampleScoringProcessor(UnsupervisedProcessor):
    """
    Wraps estimators that score any samples through ``score_samples``.

    Training samples are scored out of fold, because the estimator scores the
    samples it was fitted on as more normal than unseen ones.

    Attributes
    ----------
    estimator : SampleScorerLike
        The underlying estimator (structural type).
    """

    estimator: SampleScorerLike

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        training_scores: FloatArray = np.empty(len(X))
        for held_out in TrainingFolds.held_out_masks(len(X)):
            fold_estimator: SampleScorerLike = clone(self.estimator)
            _ = fold_estimator.fit(X[~held_out])
            training_scores[held_out] = -fold_estimator.score_samples(X[held_out])
        _ = self.estimator.fit(X)
        return training_scores

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        return -self.estimator.score_samples(X)
