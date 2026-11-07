from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from ...feature_validator import FeatureValidator
from ...parameter import AnomalyParameters
from ...types import FloatArray, NumericArray
from ..unit_normalizer import UnitNormalizer
from .processor import PrototypeSimilarityProcessor


@dataclass(frozen=True, eq=False)
class PrototypeSimilarityParameters(AnomalyParameters):
    """
    Parameters for comparing samples with reference embeddings.

    The prototypes must come from the same embedding space as the samples, such
    as text embeddings of "a normal photo" and "a damaged photo" for image
    embeddings of a joint model.

    Attributes
    ----------
    normal_prototypes : NumericArray
        Embeddings describing normal samples, with shape (n_normal, n_features).
    anomalous_prototypes : NumericArray
        Embeddings describing anomalous samples, with shape (n_anomalous, n_features).
    temperature : float
        The softmax temperature; lower values make the scores more decisive.
    decision_threshold : float
        The anomaly probability above which a sample is anomalous, in (0, 1).

    Raises
    ------
    TypeError
        If the prototypes are not numeric.
    ValueError
        If the prototypes are invalid or differ in dimension, or the temperature
        or threshold is out of range.
    """

    normal_prototypes: NumericArray
    anomalous_prototypes: NumericArray
    temperature: float = 0.01
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        FeatureValidator.require_matrix(self.normal_prototypes)
        FeatureValidator.require_matrix(self.anomalous_prototypes)
        if self.normal_prototypes.shape[1] != self.anomalous_prototypes.shape[1]:
            raise ValueError(
                "Prototypes must have the same number of features, got "
                + f"{self.normal_prototypes.shape[1]} and {self.anomalous_prototypes.shape[1]}"
            )
        if not self.temperature > 0.0:
            raise ValueError(
                f"temperature must be greater than 0, got {self.temperature}"
            )
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError(
                f"decision_threshold must be in (0, 1), got {self.decision_threshold}"
            )
        _ = self.build_processor()

    @override
    def build_processor(self) -> PrototypeSimilarityProcessor:
        """
        Build a ready-to-use prototype similarity processor.

        Returns
        -------
        PrototypeSimilarityProcessor
            The processor.

        Raises
        ------
        ValueError
            If a prototype has zero length, or the prototypes of one kind cancel out.
        """
        return PrototypeSimilarityProcessor(
            normal_prototype=self._merge(self.normal_prototypes),
            anomalous_prototype=self._merge(self.anomalous_prototypes),
            temperature=self.temperature,
            decision_threshold=self.decision_threshold,
        )

    @staticmethod
    def _merge(prototypes: NumericArray) -> FloatArray:
        mean: FloatArray = np.mean(
            UnitNormalizer.normalize(prototypes), axis=0, keepdims=True
        )
        return UnitNormalizer.normalize(mean).ravel()
