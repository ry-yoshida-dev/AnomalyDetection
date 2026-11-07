from __future__ import annotations

from typing import cast

import numpy as np

from .types import NumericArray


class FeatureValidator:
    """
    Runtime checks of feature matrices.

    The checks match those of DataScaler, so every family accepts the same inputs.
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
            If the features are not a plain numeric NumPy array. Masked arrays and
            matrices are rejected, because computations would drop their semantics.
        ValueError
            If the features are not 2-D, are empty, or contain NaN or infinity.
        """
        if not isinstance(cast(object, X), np.ndarray):
            raise TypeError(
                f"X must be a NumPy array, got {type(X).__name__}; convert it with np.asarray"
            )
        if isinstance(X, np.ma.MaskedArray | np.matrix):
            raise TypeError(
                f"X must be a plain NumPy array, got {type(X).__name__}, whose semantics would be lost"
            )
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
