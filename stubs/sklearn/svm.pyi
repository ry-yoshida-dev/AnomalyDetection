from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class OneClassSVM:
    dual_coef_: NDArray[np.floating]
    def __init__(
        self,
        *,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        nu: float = 0.5,
    ) -> None: ...
    def fit(self, X: ArrayLike) -> Self: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.floating]: ...
