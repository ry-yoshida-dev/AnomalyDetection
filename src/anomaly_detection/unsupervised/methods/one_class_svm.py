from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.svm import OneClassSVM

from ..parameter import UnsupervisedParameters
from ..processors import SampleScoringProcessor
from ..svm_gamma import SVMGamma
from ..svm_kernel import SVMKernel


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
        The kernel coefficient, or the rule that derives it.
    scaling : FeatureScaling
        Feature scaling; kernels depend on feature scale.
    """

    nu: float = 0.05
    kernel: SVMKernel = SVMKernel.RBF
    gamma: SVMGamma | float = SVMGamma.SCALE
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.nu <= 1.0:
            raise ValueError(f"nu must be in (0, 1], got {self.nu}")
        if not isinstance(self.gamma, SVMGamma):
            if not self.gamma > 0.0:
                raise ValueError(f"gamma must be greater than 0, got {self.gamma}")

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
            estimator=OneClassSVM(
                kernel=self.kernel.value,
                gamma=self.gamma.value
                if isinstance(self.gamma, SVMGamma)
                else self.gamma,
                nu=self.nu,
            ),
        )
