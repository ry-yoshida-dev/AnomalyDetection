import math
from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.ensemble import HistGradientBoostingClassifier

from ...parameter_validator import ParameterValidator
from ..parameter import SupervisedParameters
from ..processor import SupervisedProcessor


@dataclass(frozen=True)
class HistGradientBoostingParameters(SupervisedParameters):
    """
    Parameters for scikit-learn's HistGradientBoostingClassifier.

    Early stopping is off, because its validation split would hold back some of
    the few anomalous samples and fail when there are too few to stratify.

    Attributes
    ----------
    max_iter : int
        Number of boosting iterations.
    learning_rate : float
        Shrinkage applied to each tree, in (0, inf).
    max_depth : int | None
        Maximum tree depth; no limit when None.
    min_samples_leaf : int
        Minimum number of samples required at a leaf.
    random_state : int | None
        Seed for subsampling the samples that set the histogram bins, in
        [0, 2**32 - 1].

    Raises
    ------
    TypeError
        If a count or the seed is not an integer, or the learning rate is not a
        real number.
    ValueError
        If a count or the learning rate is not positive and finite, or the seed
        is out of range.
    """

    max_iter: int = 100
    learning_rate: float = 0.1
    max_depth: int | None = None
    min_samples_leaf: int = 20
    random_state: int | None = 114514

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        ParameterValidator.require_real_numbers(learning_rate=self.learning_rate)
        ParameterValidator.require_random_state(self.random_state)
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if not 0.0 < self.learning_rate < math.inf:
            raise ValueError(
                f"learning_rate must be positive and finite, got {self.learning_rate}"
            )
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 when set")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than 0")

    @override
    def build_processor(self) -> SupervisedProcessor:
        """
        Build an unfitted histogram gradient boosting processor.

        Returns
        -------
        SupervisedProcessor
            The processor.
        """
        return SupervisedProcessor(
            classifier=HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                early_stopping=False,
                class_weight="balanced" if self.is_class_balanced else None,
                random_state=self.random_state,
            ),
            scaling=FeatureScaling.NONE,
            decision_threshold=self.decision_threshold,
        )
