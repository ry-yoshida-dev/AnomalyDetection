from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.ensemble import RandomForestClassifier

from ...parameter_validator import ParameterValidator
from ..parameter import SupervisedParameters
from ..processor import SupervisedProcessor


@dataclass(frozen=True)
class RandomForestParameters(SupervisedParameters):
    """
    Parameters for scikit-learn's RandomForestClassifier.

    Attributes
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int | None
        Maximum tree depth; no limit when None.
    min_samples_leaf : int
        Minimum number of samples required at a leaf.
    n_jobs : int | None
        Number of parallel jobs.
    random_state : int | None
        Seed for bootstrapping and feature sampling.
    """

    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_leaf: int = 1
    n_jobs: int | None = None
    random_state: int | None = 114514

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_integers(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 when set")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than 0")

    @override
    def build_processor(self) -> SupervisedProcessor:
        """
        Build an unfitted random forest processor.

        Returns
        -------
        SupervisedProcessor
            The processor.
        """
        return SupervisedProcessor(
            classifier=RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced" if self.is_class_balanced else None,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            scaling=FeatureScaling.NONE,
            decision_threshold=self.decision_threshold,
        )
