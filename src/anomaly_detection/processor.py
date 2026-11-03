from __future__ import annotations

from abc import ABC, abstractmethod

from .feature_validator import FeatureValidator
from .types import BoolArray, FloatArray, NumericArray


class AnomalyProcessor(ABC):
    """
    Decides whether feature vectors are normal or anomalous.

    A higher score always means more anomalous, and a sample is anomalous when
    its score exceeds the threshold. How the scorer is prepared differs by
    family, so ``fit`` is defined by the families that need it.
    """

    def score(self, X: NumericArray) -> FloatArray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The scores with shape (n_samples,); higher is more anomalous.

        Raises
        ------
        TypeError
            If the features are not numeric.
        ValueError
            If the features are not a finite, non-empty 2-D matrix.
        NotFittedError
            If the processor requires fitting and has not been fitted.
        """
        FeatureValidator.require_matrix(X)
        return self._compute_scores(X)

    def predict(self, X: NumericArray) -> BoolArray:
        """
        Decide which samples are anomalous.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).

        Returns
        -------
        BoolArray
            True for anomalous samples, with shape (n_samples,).

        Raises
        ------
        TypeError
            If the features are not numeric.
        ValueError
            If the features are not a finite, non-empty 2-D matrix.
        NotFittedError
            If the processor requires fitting and has not been fitted.
        """
        scores: FloatArray = self.score(X)
        return scores > self._resolve_threshold(scores)

    @abstractmethod
    def _compute_scores(self, X: NumericArray) -> FloatArray: ...

    @abstractmethod
    def _resolve_threshold(self, scores: FloatArray) -> float: ...
