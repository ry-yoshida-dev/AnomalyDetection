import math
from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling

from ...parameter_validator import ParameterValidator
from ..parameter import UnsupervisedParameters
from ..sample_scoring import SampleScoringProcessor
from .gamma import SVMGamma
from .kernel import SVMKernel
from .normalized_estimator import NormalizedOneClassSVM


@dataclass(frozen=True)
class OneClassSVMParameters(UnsupervisedParameters):
    """
    Parameters for scikit-learn's OneClassSVM.

    Attributes
    ----------
    nu : float
        Upper bound on the fraction of margin errors, in (0, 1]. It shapes the
        boundary only; decisions use the contamination threshold.
    kernel : SVMKernel
        The kernel function.
    gamma : SVMGamma | float
        The kernel coefficient, positive and finite, or the rule that derives it.
    scaling : FeatureScaling
        Feature scaling; kernels depend on feature scale.

    Raises
    ------
    TypeError
        If nu or gamma is not a real number, or kernel, gamma, or scaling is not
        of its Enum.
    ValueError
        If nu is outside (0, 1] or gamma is not positive and finite.
    """

    nu: float = 0.05
    kernel: SVMKernel = SVMKernel.RBF
    gamma: SVMGamma | float = SVMGamma.SCALE
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_real_numbers(nu=self.nu)
        ParameterValidator.require_instances(SVMKernel, kernel=self.kernel)
        ParameterValidator.require_instances(FeatureScaling, scaling=self.scaling)
        if not 0.0 < self.nu <= 1.0:
            raise ValueError(f"nu must be in (0, 1], got {self.nu}")
        if not isinstance(self.gamma, SVMGamma):
            ParameterValidator.require_real_numbers(gamma=self.gamma)
            if not 0.0 < self.gamma < math.inf:
                raise ValueError(f"gamma must be positive and finite, got {self.gamma}")

    @override
    def build_processor(self) -> SampleScoringProcessor:
        """
        Build an unfitted one-class SVM processor.

        Returns
        -------
        SampleScoringProcessor
            The processor.
        """
        return SampleScoringProcessor(
            contamination=self.contamination,
            scaling=self.scaling,
            estimator=NormalizedOneClassSVM(
                kernel=self.kernel.value,
                gamma=self.gamma.value
                if isinstance(self.gamma, SVMGamma)
                else self.gamma,
                nu=self.nu,
            ),
        )
