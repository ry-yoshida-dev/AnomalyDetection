from __future__ import annotations

from dataclasses import dataclass, field
from typing import override

import numpy as np
from data_scaler import FeatureScaling, Scaler
from sklearn.exceptions import NotFittedError

from ..processor import AnomalyProcessor
from ..types import BoolArray, FloatArray, NumericArray
from .classifier import ClassifierLike


@dataclass(eq=False)
class SupervisedProcessor(AnomalyProcessor):
    """
    Detects anomalies with a binary classifier.

    The score is the estimated probability of the anomalous class.

    Attributes
    ----------
    classifier : ClassifierLike
        The underlying classifier (structural type).
    scaling : FeatureScaling
        The feature scaling, fitted on the training samples.
    decision_threshold : float
        The probability above which a sample is anomalous.
    """

    classifier: ClassifierLike
    scaling: FeatureScaling
    decision_threshold: float
    _scaler: Scaler | None = field(default=None, init=False, repr=False)

    def fit(self, X: NumericArray, y: BoolArray) -> None:
        """
        Fit the classifier.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).
        y : BoolArray
            True for anomalous samples, with shape (n_samples,).

        Raises
        ------
        TypeError
            If the features are not a plain numeric NumPy array, such as a masked
            array or a matrix, or the labels are not boolean.
        ValueError
            If the features are not a finite, non-empty 2-D matrix, their scaled
            values cannot be represented, the shapes do not match, or y does not
            contain both normal and anomalous samples.
        """
        self._scaler = None
        if not np.issubdtype(y.dtype, np.bool_):
            raise TypeError(f"y must be boolean, got dtype {y.dtype}")
        if y.ndim != 1 or y.size == 0:
            raise ValueError(f"y must be a non-empty 1-D array, got shape {y.shape}")
        if y.all() or not y.any():
            raise ValueError("y must contain both normal and anomalous samples")
        # The scaler validates the features, so the shapes are compared once they are known valid.
        scaler: Scaler = self.scaling.build_scaler()
        scaled: FloatArray = scaler.fit_transform(X)
        if y.shape != (scaled.shape[0],):
            raise ValueError(f"y must have shape ({scaled.shape[0]},), got {y.shape}")
        _ = self.classifier.fit(scaled, y)
        self._scaler = scaler

    @override
    def _validate_features(self, X: NumericArray) -> None:
        # The scaler checks the features as strictly, so a second pass would only cost time.
        pass

    @override
    def _compute_scores(self, X: NumericArray) -> FloatArray:
        if self._scaler is None:
            raise NotFittedError("Processor is not fitted yet. Call fit() first.")
        return self.classifier.predict_proba(self._scaler.transform(X))[:, 1]

    @override
    def _resolve_threshold(self, scores: FloatArray) -> float:
        return self.decision_threshold
