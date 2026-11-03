from enum import Enum


class SVMGamma(Enum):
    """
    Rule that derives the kernel coefficient of a one-class SVM.

    Attributes
    ----------
    SCALE
        The inverse of the number of features times the feature variance.
    AUTO
        The inverse of the number of features.
    """

    SCALE = "scale"
    AUTO = "auto"
