from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class IsolationForest:
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        n_jobs: int | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike) -> IsolationForest: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.floating]: ...

class RandomForestClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        class_weight: Literal["balanced"] | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> RandomForestClassifier: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]: ...

class HistGradientBoostingClassifier:
    def __init__(
        self,
        *,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        class_weight: Literal["balanced"] | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> HistGradientBoostingClassifier: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]: ...
