from enum import Enum


class NeighborsMetric(Enum):
    """
    Distance metric for neighbor-based methods.

    Attributes
    ----------
    EUCLIDEAN
        Euclidean distance.
    MANHATTAN
        Manhattan distance.
    CHEBYSHEV
        Chebyshev distance.
    COSINE
        Cosine distance.
    """

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    COSINE = "cosine"
