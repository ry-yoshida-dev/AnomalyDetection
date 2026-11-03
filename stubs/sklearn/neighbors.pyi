import numpy as np
from numpy.typing import ArrayLike, NDArray

class LocalOutlierFactor:
    negative_outlier_factor_: NDArray[np.floating]

    def __init__(
        self,
        n_neighbors: int = 20,
        *,
        metric: str = "minkowski",
        novelty: bool = False,
        n_jobs: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike) -> LocalOutlierFactor: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.floating]: ...

class NearestNeighbors:
    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike) -> NearestNeighbors: ...
    def kneighbors(
        self, X: ArrayLike | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.intp]]: ...
