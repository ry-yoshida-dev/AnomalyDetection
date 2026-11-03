"""Structural type for classifiers used by SupervisedProcessor."""

from __future__ import annotations

from typing import Protocol

from ..types import BoolArray, FloatArray, NumericArray


class ClassifierLike(Protocol):
    """
    Structural type for classifiers that estimate class probabilities.

    Any scikit-learn compatible classifier fits. Probability columns must follow
    the sorted order of the labels, so the last one is the anomalous class.
    """

    def fit(self, X: NumericArray, y: BoolArray) -> object:
        """
        Fit the classifier.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : BoolArray
            True for anomalous samples.

        Returns
        -------
        object
            The fitted classifier.
        """
        ...

    def predict_proba(self, X: NumericArray) -> FloatArray:
        """
        Estimate class probabilities.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        FloatArray
            The probabilities with shape (n_samples, 2).
        """
        ...
