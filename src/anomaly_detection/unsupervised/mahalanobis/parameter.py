from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.covariance import LedoitWolf

from ...parameter_validator import ParameterValidator
from ..parameter import UnsupervisedParameters
from .processor import MahalanobisProcessor


@dataclass(frozen=True)
class MahalanobisParameters(UnsupervisedParameters):
    """
    Parameters for scoring by the Mahalanobis distance from the training mean.

    Covariance is estimated by Ledoit-Wolf shrinkage, which stays invertible
    when features outnumber samples.

    Attributes
    ----------
    scaling : FeatureScaling
        Feature scaling; shrinkage depends on feature scale.

    Raises
    ------
    TypeError
        If scaling is not a FeatureScaling.
    """

    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_instances(FeatureScaling, scaling=self.scaling)

    @override
    def build_processor(self) -> MahalanobisProcessor:
        """
        Build an unfitted Mahalanobis processor.

        Returns
        -------
        MahalanobisProcessor
            The processor.
        """
        return MahalanobisProcessor(
            contamination=self.contamination,
            scaling=self.scaling,
            estimator=LedoitWolf(),
        )
