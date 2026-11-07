from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class GaussianMixture:
    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        n_init: int = 1,
        random_state: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike) -> Self: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.floating]: ...
