from __future__ import annotations

import numpy as np

from ..types import FloatArray, NumericArray


class UnitNormalizer:
    """
    Scales row vectors to unit length for cosine similarity.
    """

    @staticmethod
    def normalize(X: NumericArray) -> FloatArray:
        """
        Scale each row to unit length.

        Parameters
        ----------
        X : NumericArray
            The row vectors with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The unit vectors with the same shape, in float64.

        Raises
        ------
        ValueError
            If a row has zero length and therefore no direction.
        """
        values: FloatArray = np.asarray(X, dtype=np.float64)
        # Dividing by the largest magnitude first keeps the squares from overflowing or underflowing.
        peaks: FloatArray = np.abs(values).max(axis=1, keepdims=True)
        if not peaks.all():
            raise ValueError("Vectors must not have zero length")
        bounded: FloatArray = values / peaks
        return bounded / np.linalg.norm(bounded, axis=1, keepdims=True)
