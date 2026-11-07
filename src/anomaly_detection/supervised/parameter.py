from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

from ..parameter import AnomalyParameters
from ..parameter_validator import ParameterValidator
from .processor import SupervisedProcessor


@dataclass(frozen=True, kw_only=True)
class SupervisedParameters(AnomalyParameters, ABC):
    """
    Base class for parameters of supervised methods.

    Attributes
    ----------
    decision_threshold : float
        The anomaly probability above which a sample is anomalous, in (0, 1).
    is_class_balanced : bool
        Whether samples are weighted inversely to their class frequency, which
        keeps rare anomalies from being ignored.

    Raises
    ------
    TypeError
        If the threshold is not a real number or is_class_balanced is not a bool.
    ValueError
        If the threshold is outside (0, 1).
    """

    decision_threshold: float = 0.5
    is_class_balanced: bool = True

    def __post_init__(self) -> None:
        ParameterValidator.require_real_numbers(
            decision_threshold=self.decision_threshold
        )
        ParameterValidator.require_instances(
            bool, is_class_balanced=self.is_class_balanced
        )
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError(
                f"decision_threshold must be in (0, 1), got {self.decision_threshold}"
            )

    @abstractmethod
    @override
    def build_processor(self) -> SupervisedProcessor:
        """
        Build an unfitted supervised processor.

        Returns
        -------
        SupervisedProcessor
            The processor.
        """
