from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

import numpy as np
from data_scaler import FeatureScaling, Scaler
from sklearn.exceptions import NotFittedError

from ..processor import AnomalyProcessor
from ..types import FloatArray, NumericArray


@dataclass(eq=False)
class UnsupervisedProcessor(AnomalyProcessor, ABC):
    """
    Learns normality from unlabeled samples and sets the threshold from them.

    The threshold is the quantile of the training scores that leaves the
    contamination fraction above it, so all methods share one meaning of
    contamination regardless of their score scale.

    Attributes
    ----------
    contamination : float
        The expected fraction of anomalous training samples.
    scaling : FeatureScaling
        The feature scaling, fitted on the training samples.
    """

    contamination: float
    scaling: FeatureScaling
    _scaler: Scaler | None = field(default=None, init=False, repr=False)
    _threshold: float | None = field(default=None, init=False, repr=False)

    def fit(self, X: NumericArray) -> None:
        """
        Fit the scorer and set the threshold.

        Parameters
        ----------
        X : NumericArray
            The unlabeled feature matrix with shape (n_samples, n_features).

        Raises
        ------
        TypeError
            If the features are not a plain numeric NumPy array, such as a masked
            array or a matrix.
        ValueError
            If the features are not a finite, non-empty 2-D matrix, their scaled
            values cannot be represented, or there are too few samples for the
            method.
        """
        self._scaler = None
        self._threshold = None
        scaler: Scaler = self.scaling.build_scaler()
        training_scores: FloatArray = self._fit_scorer(scaler.fit_transform(X))
        self._threshold = float(np.quantile(training_scores, 1.0 - self.contamination))
        self._scaler = scaler

    @property
    def threshold(self) -> float:
        """
        Get the score above which a sample is anomalous.

        Returns
        -------
        float
            The threshold set by the last fit.

        Raises
        ------
        NotFittedError
            If fit has not completed successfully yet.
        """
        if self._threshold is None:
            raise NotFittedError("Processor is not fitted yet. Call fit() first.")
        return self._threshold

    @override
    def _validate_features(self, X: NumericArray) -> None:
        # The scaler checks the features as strictly, so a second pass would only cost time.
        pass

    @override
    def _compute_scores(self, X: NumericArray) -> FloatArray:
        if self._scaler is None:
            raise NotFittedError("Processor is not fitted yet. Call fit() first.")
        return self._score_scaled(self._scaler.transform(X))

    @override
    def _resolve_threshold(self, scores: FloatArray) -> float:
        return self.threshold

    @abstractmethod
    def _fit_scorer(self, X: FloatArray) -> FloatArray:
        """Fit on scaled samples and return their scores, excluding each sample from its own reference."""

    @abstractmethod
    def _score_scaled(self, X: FloatArray) -> FloatArray: ...
