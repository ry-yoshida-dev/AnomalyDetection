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
            The unit vectors with the same shape.

        Raises
        ------
        ValueError
            If a row has zero length and therefore no direction.
        """
        norms: FloatArray = np.sqrt(np.square(X).sum(axis=1, keepdims=True))
        if not norms.all():
            raise ValueError("Vectors must not have zero length")
        return X / norms
