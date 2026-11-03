from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class LogisticRegression:
    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 100,
        class_weight: Literal["balanced"] | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]: ...
