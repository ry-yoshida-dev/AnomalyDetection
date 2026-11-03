from __future__ import annotations

import numpy as np

from .types import NumericArray


class FeatureValidator:
    """
    Runtime checks of feature matrices.
    """

    @staticmethod
    def require_matrix(X: NumericArray) -> None:
        """
        Check that features form a finite, non-empty numeric matrix.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Raises
        ------
        TypeError
            If the features are not numeric.
        ValueError
            If the features are not 2-D, are empty, or contain NaN or infinity.
        """
        is_numeric: bool = np.issubdtype(X.dtype, np.integer) or np.issubdtype(
            X.dtype, np.floating
        )
        if not is_numeric:
            raise TypeError(f"X must be numeric, got dtype {X.dtype}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim} dimensions")
        if X.size == 0:
            raise ValueError(f"X must not be empty, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinity")
