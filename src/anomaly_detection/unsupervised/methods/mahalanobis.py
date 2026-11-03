from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.covariance import LedoitWolf

from ..parameter import UnsupervisedParameters
from ..processors import MahalanobisProcessor


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
    """

    scaling: FeatureScaling = FeatureScaling.STANDARD

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
