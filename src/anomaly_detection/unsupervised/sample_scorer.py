"""Structural type for estimators used by SampleScoringProcessor."""

from __future__ import annotations

from typing import Protocol

from ..types import FloatArray, NumericArray


class SampleScorerLike(Protocol):
    """
    Structural type for outlier estimators whose higher scores mean more normal.
    """

    def fit(self, X: NumericArray) -> object:
        """
        Fit the estimator.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        object
            The fitted estimator.
        """
        ...

    def score_samples(self, X: NumericArray) -> FloatArray:
        """
        Compute normality scores.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        FloatArray
            The scores with shape (n_samples,); higher is more normal.
        """
        ...
