from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.ensemble import IsolationForest

from ...parameter_validator import ParameterValidator
from ..parameter import UnsupervisedParameters
from ..processors import SampleScoringProcessor


@dataclass(frozen=True)
class IsolationForestParameters(UnsupervisedParameters):
    """
    Parameters for scikit-learn's IsolationForest.

    Attributes
    ----------
    n_estimators : int
        Number of trees.
    max_samples : int | None
        Samples drawn per tree; at most 256 when None.
    n_jobs : int | None
        Number of parallel jobs.
    random_state : int | None
        Seed for subsampling and splits.
    """

    n_estimators: int = 100
    max_samples: int | None = None
    n_jobs: int | None = None
    random_state: int | None = 114514

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_integers(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be greater than 0 when set")

    @override
    def build_processor(self) -> SampleScoringProcessor:
        """
        Build an unfitted isolation forest processor.

        Returns
        -------
        SampleScoringProcessor
            The processor.
        """
        return SampleScoringProcessor(
            contamination=self.contamination,
            scaling=FeatureScaling.NONE,
            estimator=IsolationForest(
                n_estimators=self.n_estimators,
                max_samples="auto" if self.max_samples is None else self.max_samples,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
        )
