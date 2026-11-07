from __future__ import annotations

from dataclasses import dataclass, field
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from ...types import FloatArray
from ..processor import UnsupervisedProcessor


@dataclass(eq=False)
class EnsembleProcessor(UnsupervisedProcessor):
    """
    Averages the score percentiles of several unsupervised processors.

    Each member's score is replaced by its percentile among that member's
    out-of-fold training scores, which puts every member on the same scale
    from 0 to 1. Percentiles saturate at 1 beyond the largest training score,
    so samples that every member places beyond all training samples tie.

    Attributes
    ----------
    members : tuple[UnsupervisedProcessor, ...]
        The unfitted member processors; each applies its own scaling, and their
        thresholds are not used.
    """

    members: tuple[UnsupervisedProcessor, ...]
    _sorted_member_scores: tuple[FloatArray, ...] | None = field(
        default=None, init=False, repr=False
    )

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        self._sorted_member_scores = None
        member_scores: list[FloatArray] = [
            member._fit_with_training_scores(X) for member in self.members
        ]
        sorted_member_scores: tuple[FloatArray, ...] = tuple(
            np.sort(scores) for scores in member_scores
        )
        training_scores: FloatArray = self._average_percentiles(
            member_scores, sorted_member_scores
        )
        self._sorted_member_scores = sorted_member_scores
        return training_scores

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        if self._sorted_member_scores is None:
            raise NotFittedError("Processor is not fitted yet. Call fit() first.")
        member_scores: list[FloatArray] = [member.score(X) for member in self.members]
        return self._average_percentiles(member_scores, self._sorted_member_scores)

    @staticmethod
    def _average_percentiles(
        member_scores: list[FloatArray], sorted_member_scores: tuple[FloatArray, ...]
    ) -> FloatArray:
        """Average each member's percentiles, counting ties as half below."""
        percentiles: FloatArray = np.empty((len(member_scores), len(member_scores[0])))
        for index, (scores, references) in enumerate(
            zip(member_scores, sorted_member_scores, strict=True)
        ):
            below: NDArray[np.intp] = np.searchsorted(references, scores, side="left")
            at_most: NDArray[np.intp] = np.searchsorted(
                references, scores, side="right"
            )
            percentiles[index] = (below + at_most) / (2 * len(references))
        averaged: FloatArray = percentiles.mean(axis=0)
        return averaged
