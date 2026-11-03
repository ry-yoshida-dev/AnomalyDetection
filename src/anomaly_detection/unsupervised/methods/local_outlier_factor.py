from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.neighbors import LocalOutlierFactor

from ...neighbors_metric import NeighborsMetric
from ...parameter_validator import ParameterValidator
from ..parameter import UnsupervisedParameters
from ..processors import LocalOutlierFactorProcessor


@dataclass(frozen=True)
class LocalOutlierFactorParameters(UnsupervisedParameters):
    """
    Parameters for scikit-learn's LocalOutlierFactor in novelty mode.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors defining the local density.
    metric : NeighborsMetric
        The distance metric.
    scaling : FeatureScaling
        Feature scaling; distances depend on feature scale.
    n_jobs : int | None
        Number of parallel jobs.
    """

    n_neighbors: int = 20
    metric: NeighborsMetric = NeighborsMetric.EUCLIDEAN
    scaling: FeatureScaling = FeatureScaling.STANDARD
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_integers(
            n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
        )
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than 0")

    @override
    def build_processor(self) -> LocalOutlierFactorProcessor:
        """
        Build an unfitted local outlier factor processor.

        Returns
        -------
        LocalOutlierFactorProcessor
            The processor.
        """
        return LocalOutlierFactorProcessor(
            contamination=self.contamination,
            scaling=self.scaling,
            estimator=LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                metric=self.metric.value,
                novelty=True,
                n_jobs=self.n_jobs,
            ),
        )
