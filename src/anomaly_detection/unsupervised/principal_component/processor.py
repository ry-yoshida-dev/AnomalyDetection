from __future__ import annotations

from dataclasses import dataclass, field
from typing import override

import numpy as np
from sklearn.exceptions import NotFittedError

from ...types import FloatArray
from ..processor import UnsupervisedProcessor
from ..training_folds import TrainingFolds
from .principal_subspace import PrincipalSubspace


@dataclass(eq=False)
class PrincipalComponentProcessor(UnsupervisedProcessor):
    """
    Scores samples by their squared distance from the principal subspace.

    Training samples are scored out of fold, because the subspace they helped
    estimate lies closer to them than to unseen samples.

    Attributes
    ----------
    n_components : int | None
        The number of components kept; chosen by explained variance when None.
    explained_variance : float
        The fraction of variance the kept components explain when n_components
        is None.
    """

    n_components: int | None
    explained_variance: float
    _subspace: PrincipalSubspace | None = field(default=None, init=False, repr=False)

    @override
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        self._subspace = None
        training_scores: FloatArray = np.empty(len(X))
        for held_out in TrainingFolds.held_out_masks(len(X)):
            fold_subspace: PrincipalSubspace = PrincipalSubspace.fit(
                X[~held_out], self.n_components, self.explained_variance
            )
            training_scores[held_out] = fold_subspace.residuals(X[held_out])
        self._subspace = PrincipalSubspace.fit(
            X, self.n_components, self.explained_variance
        )
        return training_scores

    @override
    def _score_scaled(self, X: FloatArray) -> FloatArray:
        if self._subspace is None:
            raise NotFittedError("Processor is not fitted yet. Call fit() first.")
        return self._subspace.residuals(X)
