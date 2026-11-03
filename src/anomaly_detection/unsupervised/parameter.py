from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

from ..parameter import AnomalyParameters
from .processor import UnsupervisedProcessor


@dataclass(frozen=True, kw_only=True)
class UnsupervisedParameters(AnomalyParameters, ABC):
    """
    Base class for parameters of unsupervised methods.

    Attributes
    ----------
    contamination : float
        The expected fraction of anomalous training samples, in (0, 1). With
        purely normal training samples, it approximates the false positive rate.

    Raises
    ------
    ValueError
        If contamination is outside (0, 1).
    """

    contamination: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination < 1.0:
            raise ValueError(
                f"contamination must be in (0, 1), got {self.contamination}"
            )

    @abstractmethod
    @override
    def build_processor(self) -> UnsupervisedProcessor:
        """
        Build an unfitted unsupervised processor.

        Returns
        -------
        UnsupervisedProcessor
            The processor.
        """
