from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling

from ..parameter import UnsupervisedParameters
from .processor import EmpiricalCumulativeDistributionProcessor


@dataclass(frozen=True)
class EmpiricalCumulativeDistributionParameters(UnsupervisedParameters):
    """
    Parameters for ECOD, scoring by tail probabilities of each feature.

    It needs no hyperparameters and is fast, which makes it a good first method
    to try. Features are treated independently, so anomalies that only break
    correlations between features go unnoticed.
    """

    @override
    def build_processor(self) -> EmpiricalCumulativeDistributionProcessor:
        """
        Build an unfitted ECOD processor.

        Returns
        -------
        EmpiricalCumulativeDistributionProcessor
            The processor.
        """
        # Tail probabilities depend only on the order of values, which scaling keeps.
        return EmpiricalCumulativeDistributionProcessor(
            contamination=self.contamination,
            scaling=FeatureScaling.NONE,
        )
