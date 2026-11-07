from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling

from ...parameter_validator import ParameterValidator
from ..parameter import UnsupervisedParameters
from .processor import PrincipalComponentProcessor


@dataclass(frozen=True)
class PrincipalComponentParameters(UnsupervisedParameters):
    """
    Parameters for scoring by the reconstruction error from principal components.

    Suits high-dimensional features such as embeddings, whose normal samples
    vary along few directions. At least one direction is always left out, so
    features need at least two dimensions.

    Attributes
    ----------
    n_components : int | None
        The number of components kept, below the number of features. When None,
        the fewest components explaining ``explained_variance`` are kept.
    explained_variance : float
        The fraction of variance to explain when n_components is None, in (0, 1).
    scaling : FeatureScaling
        Feature scaling; principal components depend on feature scale.

    Raises
    ------
    TypeError
        If n_components is not an integer, explained_variance is not a real
        number, or scaling is not a FeatureScaling.
    ValueError
        If n_components is not positive or explained_variance is outside (0, 1).
    """

    n_components: int | None = None
    explained_variance: float = 0.9
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        super().__post_init__()
        ParameterValidator.require_integers(n_components=self.n_components)
        ParameterValidator.require_real_numbers(
            explained_variance=self.explained_variance
        )
        ParameterValidator.require_instances(FeatureScaling, scaling=self.scaling)
        if self.n_components is not None and self.n_components <= 0:
            raise ValueError(
                f"n_components must be greater than 0 when set, got {self.n_components}"
            )
        if not 0.0 < self.explained_variance < 1.0:
            raise ValueError(
                f"explained_variance must be in (0, 1), got {self.explained_variance}"
            )

    @override
    def build_processor(self) -> PrincipalComponentProcessor:
        """
        Build an unfitted principal component processor.

        Returns
        -------
        PrincipalComponentProcessor
            The processor.
        """
        return PrincipalComponentProcessor(
            contamination=self.contamination,
            scaling=self.scaling,
            n_components=self.n_components,
            explained_variance=self.explained_variance,
        )
