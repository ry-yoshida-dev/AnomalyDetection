import numpy as np
from numpy.typing import ArrayLike, NDArray

class OneClassSVM:
    def __init__(
        self,
        *,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        nu: float = 0.5,
    ) -> None: ...
    def fit(self, X: ArrayLike) -> OneClassSVM: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.floating]: ...
