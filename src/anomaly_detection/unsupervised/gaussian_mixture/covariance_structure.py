from enum import Enum


class CovarianceStructure(Enum):
    """
    Covariance each component of a Gaussian mixture may have.

    Attributes
    ----------
    FULL
        Each component has its own unrestricted covariance.
    TIED
        All components share one unrestricted covariance.
    DIAGONAL
        Each component has its own diagonal covariance, for many features.
    SPHERICAL
        Each component has its own single variance.
    """

    FULL = "full"
    TIED = "tied"
    DIAGONAL = "diag"
    SPHERICAL = "spherical"
