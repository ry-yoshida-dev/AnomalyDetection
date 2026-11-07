from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.neighbors import NearestNeighbors

from ...neighbors_metric import NeighborsMetric
from ...parameter_validator import ParameterValidator
from ..parameter import UnsupervisedParameters
from .processor import KNearestNeighborsProcessor


@dataclass(frozen=True)
class KNearestNeighborsParameters(UnsupervisedParameters):
    """
    Parameters for scoring by the distance to the nearest training samples.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors averaged; must be below the number of samples.
    metric : NeighborsMetric
        The distance metric.
    scaling : FeatureScaling
        Feature scaling; distances depend on feature scale.
    n_jobs : int | None
        Number of parallel jobs.
    """

    n_neighbors: int = 5
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
    def build_processor(self) -> KNearestNeighborsProcessor:
        """
        Build an unfitted k-nearest neighbors processor.

        Returns
        -------
        KNearestNeighborsProcessor
            The processor.
        """
        return KNearestNeighborsProcessor(
            contamination=self.contamination,
            scaling=self.scaling,
            estimator=NearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric=self.metric.value,
                n_jobs=self.n_jobs,
            ),
        )
