from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import AnomalyProcessor


class AnomalyParameters(ABC):
    """
    Base class for anomaly detection parameters that build a processor.
    """

    @abstractmethod
    def build_processor(self) -> AnomalyProcessor:
        """
        Build an unfitted anomaly processor.

        Returns
        -------
        AnomalyProcessor
            The processor.
        """
