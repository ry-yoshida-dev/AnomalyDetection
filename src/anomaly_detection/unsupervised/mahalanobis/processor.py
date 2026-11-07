from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from sklearn.base import clone
from sklearn.covariance import LedoitWolf

from ...types import FloatArray
from ..processor import UnsupervisedProcessor
from ..training_folds import TrainingFolds


@dataclass(eq=False)
class MahalanobisProcessor(UnsupervisedProcessor):
    """
    Scores samples by their squared Mahalanobis distance from the training mean.

    Training samples are scored out of fold, because the mean and covariance
    they helped estimate place them closer than unseen samples.

    Attributes
    ----------
    estimator : LedoitWolf
        The underlying shrinkage covariance estimator.
    """

    estimator: LedoitWolf

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        training_scores: FloatArray = np.empty(len(X))
        for held_out in TrainingFolds.held_out_masks(len(X)):
            fold_estimator: LedoitWolf = clone(self.estimator).fit(X[~held_out])
            training_scores[held_out] = fold_estimator.mahalanobis(X[held_out])
        _ = self.estimator.fit(X)
        return training_scores

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        return self.estimator.mahalanobis(X)
