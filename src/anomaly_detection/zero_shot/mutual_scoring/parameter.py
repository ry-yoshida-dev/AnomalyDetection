import math
from dataclasses import dataclass
from typing import override

from ...neighbors_metric import NeighborsMetric
from ...parameter import AnomalyParameters
from ...parameter_validator import ParameterValidator
from .processor import MutualScoringProcessor


@dataclass(frozen=True)
class MutualScoringParameters(AnomalyParameters):
    """
    Parameters for scoring samples against the rest of their batch.

    Attributes
    ----------
    n_neighbors : int
        Number of other samples averaged; batches must be larger than this.
    metric : NeighborsMetric
        The distance metric.
    deviation_threshold : float
        Scaled median absolute deviations above the median score beyond which a
        sample is anomalous, positive and finite.
    n_jobs : int | None
        Number of parallel jobs.

    Raises
    ------
    TypeError
        If a count is not an integer, the deviation threshold is not a real
        number, or metric is not a NeighborsMetric.
    ValueError
        If n_neighbors is not positive, n_jobs is 0, or the deviation threshold
        is not positive and finite.
    """

    n_neighbors: int = 5
    metric: NeighborsMetric = NeighborsMetric.COSINE
    deviation_threshold: float = 3.5
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
        )
        ParameterValidator.require_real_numbers(
            deviation_threshold=self.deviation_threshold
        )
        ParameterValidator.require_instances(NeighborsMetric, metric=self.metric)
        ParameterValidator.require_job_count(self.n_jobs)
        if self.n_neighbors <= 0:
            raise ValueError(
                f"n_neighbors must be greater than 0, got {self.n_neighbors}"
            )
        if not 0.0 < self.deviation_threshold < math.inf:
            raise ValueError(
                f"deviation_threshold must be positive and finite, got {self.deviation_threshold}"
            )

    @override
    def build_processor(self) -> MutualScoringProcessor:
        """
        Build a ready-to-use mutual scoring processor.

        Returns
        -------
        MutualScoringProcessor
            The processor.
        """
        return MutualScoringProcessor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            deviation_threshold=self.deviation_threshold,
            n_jobs=self.n_jobs,
        )
