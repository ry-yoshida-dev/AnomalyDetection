from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.linear_model import LogisticRegression

from ...parameter_validator import ParameterValidator
from ..parameter import SupervisedParameters
from ..processor import SupervisedProcessor


@dataclass(frozen=True)
class LogisticRegressionParameters(SupervisedParameters):
    """
    Parameters for scikit-learn's LogisticRegression.

    Attributes
    ----------
    C : float
        Inverse regularization strength, in (0, inf]; inf turns regularization off.
    max_iter : int
        Maximum number of solver iterations.
    scaling : FeatureScaling
        Feature scaling; regularization and solvers depend on feature scale.

    Raises
    ------
    TypeError
        If C is not a real number, max_iter is not an integer, or scaling is
        not a FeatureScaling.
    ValueError
        If C or max_iter is not positive.
    """

    C: float = 1.0
    max_iter: int = 100
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_integers(max_iter=self.max_iter)
        ParameterValidator.require_real_numbers(C=self.C)
        ParameterValidator.require_instances(FeatureScaling, scaling=self.scaling)
        if not self.C > 0.0:
            raise ValueError(f"C must be greater than 0, got {self.C}")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")

    @override
    def build_processor(self) -> SupervisedProcessor:
        """
        Build an unfitted logistic regression processor.

        Returns
        -------
        SupervisedProcessor
            The processor.
        """
        return SupervisedProcessor(
            classifier=LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight="balanced" if self.is_class_balanced else None,
            ),
            scaling=self.scaling,
            decision_threshold=self.decision_threshold,
        )
