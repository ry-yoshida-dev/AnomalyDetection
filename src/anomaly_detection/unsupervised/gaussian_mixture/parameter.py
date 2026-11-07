from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.mixture import GaussianMixture

from ...parameter_validator import ParameterValidator
from ..parameter import UnsupervisedParameters
from ..sample_scoring import SampleScoringProcessor
from .covariance_structure import CovarianceStructure


@dataclass(frozen=True)
class GaussianMixtureParameters(UnsupervisedParameters):
    """
    Parameters for scoring by the negative log-likelihood of a Gaussian mixture.

    Attributes
    ----------
    n_components : int
        Number of mixture components; at most the number of training samples
        in each of the five folds.
    covariance : CovarianceStructure
        The covariance each component may have. DIAGONAL or SPHERICAL keeps
        fitting tractable for many features.
    n_init : int
        Number of initializations, of which the most likely fit is kept.
    random_state : int | None
        Seed for initialization, in [0, 2**32 - 1].
    scaling : FeatureScaling
        Feature scaling; restricted covariances depend on feature scale.

    Raises
    ------
    TypeError
        If a count or the seed is not an integer, or covariance or scaling is
        not of its Enum.
    ValueError
        If a count is not positive or the seed is out of range.
    """

    n_components: int = 2
    covariance: CovarianceStructure = CovarianceStructure.FULL
    n_init: int = 1
    random_state: int | None = 114514
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_integers(
            n_components=self.n_components,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        ParameterValidator.require_instances(
            CovarianceStructure, covariance=self.covariance
        )
        ParameterValidator.require_instances(FeatureScaling, scaling=self.scaling)
        ParameterValidator.require_random_state(self.random_state)
        if self.n_components <= 0:
            raise ValueError(
                f"n_components must be greater than 0, got {self.n_components}"
            )
        if self.n_init <= 0:
            raise ValueError(f"n_init must be greater than 0, got {self.n_init}")

    @override
    def build_processor(self) -> SampleScoringProcessor:
        """
        Build an unfitted Gaussian mixture processor.

        Returns
        -------
        SampleScoringProcessor
            The processor.
        """
        return SampleScoringProcessor(
            contamination=self.contamination,
            scaling=self.scaling,
            estimator=GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance.value,
                n_init=self.n_init,
                random_state=self.random_state,
            ),
        )
