from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, override

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from ...types import BoolArray, FloatArray
from ..processor import UnsupervisedProcessor


@dataclass(eq=False)
class EmpiricalCumulativeDistributionProcessor(UnsupervisedProcessor):
    """
    Scores samples by the improbability of their feature values' tails.

    Per feature, the tail probabilities are the fractions of training values at
    most and at least the sample's value. The score is the largest of three
    sums of negative log tail probabilities over the features: of the left
    tails, of the right tails, and of the tail each feature is skewed toward.

    Tail probabilities count the sample as one of the references, which keeps
    them above zero; training samples are counted against the other training
    samples, so their scores are distributed as those of unseen samples.
    """

    _sorted_training: FloatArray | None = field(default=None, init=False, repr=False)
    _is_left_skewed: BoolArray | None = field(default=None, init=False, repr=False)

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        self._sorted_training = None
        n_samples: int = len(X)
        sorted_training: FloatArray = np.sort(X, axis=0)
        centered: FloatArray = X - X.mean(axis=0)
        is_left_skewed: BoolArray = (centered**3).mean(axis=0) < 0.0
        # Each count includes the sample itself, which stands in for the added reference.
        at_most: NDArray[np.intp] = self._count(sorted_training, X, "right")
        at_least: NDArray[np.intp] = n_samples - self._count(sorted_training, X, "left")
        training_scores: FloatArray = self._combine(
            at_most / n_samples, at_least / n_samples, is_left_skewed
        )
        self._sorted_training = sorted_training
        self._is_left_skewed = is_left_skewed
        return training_scores

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        if self._sorted_training is None or self._is_left_skewed is None:
            raise NotFittedError("Processor is not fitted yet. Call fit() first.")
        n_training: int = len(self._sorted_training)
        at_most: NDArray[np.intp] = self._count(self._sorted_training, X, "right")
        at_least: NDArray[np.intp] = n_training - self._count(
            self._sorted_training, X, "left"
        )
        return self._combine(
            (at_most + 1) / (n_training + 1),
            (at_least + 1) / (n_training + 1),
            self._is_left_skewed,
        )

    @staticmethod
    def _count(
        sorted_training: FloatArray, X: FloatArray, side: Literal["left", "right"]
    ) -> NDArray[np.intp]:
        """Count training values below each value, including equal ones when side is right."""
        counts: NDArray[np.intp] = np.empty(X.shape, dtype=np.intp)
        for feature in range(np.size(X, axis=1)):
            counts[:, feature] = np.searchsorted(
                sorted_training[:, feature], X[:, feature], side=side
            )
        return counts

    @staticmethod
    def _combine(
        left_tail: FloatArray, right_tail: FloatArray, is_left_skewed: BoolArray
    ) -> FloatArray:
        """Take the largest of the left, right, and skewness-chosen tail scores."""
        left_scores: FloatArray = -np.log(left_tail)
        right_scores: FloatArray = -np.log(right_tail)
        skewed_scores: FloatArray = np.where(is_left_skewed, left_scores, right_scores)
        largest: FloatArray = np.maximum(
            np.maximum(left_scores.sum(axis=1), right_scores.sum(axis=1)),
            skewed_scores.sum(axis=1),
        )
        return largest
