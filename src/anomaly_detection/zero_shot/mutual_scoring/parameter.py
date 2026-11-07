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
        sample is anomalous.
    n_jobs : int | None
        Number of parallel jobs.
    """

    n_neighbors: int = 5
    metric: NeighborsMetric = NeighborsMetric.COSINE
    deviation_threshold: float = 3.5
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
        )
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than 0")
        if not self.deviation_threshold > 0.0:
            raise ValueError(
                f"deviation_threshold must be greater than 0, got {self.deviation_threshold}"
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
