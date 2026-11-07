from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling, ScalingKind
from sklearn.neighbors import LocalOutlierFactor

from ...neighbors_metric import NeighborsMetric
from ...parameter_validator import ParameterValidator
from ..parameter import UnsupervisedParameters
from .processor import LocalOutlierFactorProcessor


@dataclass(frozen=True)
class LocalOutlierFactorParameters(UnsupervisedParameters):
    """
    Parameters for scikit-learn's LocalOutlierFactor in novelty mode.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors defining the local density.
    metric : NeighborsMetric
        The distance metric. Cosine distance needs a scaling that keeps the
        origin, such as NONE, MAX_ABS, or UNIT_NORM, because shifting moves
        typical samples to the zero vector, which has no direction.
    scaling : FeatureScaling
        Feature scaling; distances depend on feature scale.
    n_jobs : int | None
        Number of parallel jobs.

    Raises
    ------
    TypeError
        If a count is not an integer, or metric or scaling is not of its Enum.
    ValueError
        If n_neighbors is not positive, n_jobs is 0, or cosine distance is
        combined with a scaling that shifts the features.
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
        ParameterValidator.require_instances(NeighborsMetric, metric=self.metric)
        ParameterValidator.require_instances(FeatureScaling, scaling=self.scaling)
        ParameterValidator.require_job_count(self.n_jobs)
        if self.n_neighbors <= 0:
            raise ValueError(
                f"n_neighbors must be greater than 0, got {self.n_neighbors}"
            )
        is_origin_moved: bool = self.scaling.kind in (
            ScalingKind.AFFINE,
            ScalingKind.MONOTONIC,
        )
        if self.metric is NeighborsMetric.COSINE and is_origin_moved:
            raise ValueError(
                f"Cosine distance needs a scaling that keeps the origin, got {self.scaling}"
            )

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
