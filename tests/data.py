"""Shared synthetic features for the tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SyntheticFeatures:
    """
    Normal samples around the origin and anomalous samples far from them.
    """

    @staticmethod
    def normal(n_samples: int = 200, seed: int = 0) -> NDArray[np.float64]:
        """
        Draw normal samples.

        Parameters
        ----------
        n_samples : int
            The number of samples.
        seed : int
            The random seed.

        Returns
        -------
        NDArray[np.float64]
            The samples with shape (n_samples, 4).
        """
        return np.random.default_rng(seed).standard_normal((n_samples, 4))

    @staticmethod
    def anomalous(n_samples: int = 10, seed: int = 1) -> NDArray[np.float64]:
        """
        Draw anomalous samples.

        Parameters
        ----------
        n_samples : int
            The number of samples.
        seed : int
            The random seed.

        Returns
        -------
        NDArray[np.float64]
            The samples with shape (n_samples, 4).
        """
        return np.random.default_rng(seed).standard_normal((n_samples, 4)) + 12.0
