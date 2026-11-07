from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...types import FloatArray


@dataclass(frozen=True)
class PrincipalSubspace:
    """
    The mean and leading principal components of a set of samples.

    Attributes
    ----------
    mean : FloatArray
        The sample mean with shape (n_features,).
    components : FloatArray
        Orthonormal components with shape (n_components, n_features).
    """

    mean: FloatArray
    components: FloatArray

    @classmethod
    def fit(
        cls, X: FloatArray, n_components: int | None, explained_variance: float
    ) -> PrincipalSubspace:
        """
        Find the principal subspace of samples.

        Parameters
        ----------
        X : FloatArray
            The samples with shape (n_samples, n_features), at least 2 features.
        n_components : int | None
            The number of components kept, below the number of features; when
            None, the fewest explaining ``explained_variance`` are kept.
        explained_variance : float
            The fraction of variance to explain when n_components is None.

        Returns
        -------
        PrincipalSubspace
            The subspace, always leaving at least one direction out.

        Raises
        ------
        ValueError
            If there is only one feature, or n_components is not below the
            number of features.
        """
        n_features: int = np.size(X, axis=1)
        if n_features < 2:
            raise ValueError(
                "Principal component residuals need at least 2 features, got 1"
            )
        if n_components is not None and n_components >= n_features:
            raise ValueError(
                f"n_components must be below the number of features ({n_features}), got {n_components}"
            )
        mean: FloatArray = X.mean(axis=0)
        singular_values: FloatArray
        right_vectors: FloatArray
        _, singular_values, right_vectors = np.linalg.svd(X - mean, full_matrices=False)
        n_kept: int
        if n_components is not None:
            n_kept = n_components
        else:
            variances: FloatArray = singular_values**2
            total_variance: float = float(variances.sum())
            if total_variance == 0.0:
                n_kept = 1
            else:
                cumulative: FloatArray = np.cumsum(variances) / total_variance
                n_kept = int(np.searchsorted(cumulative, explained_variance)) + 1
        # Keeping every direction would leave no residual to score.
        n_kept = min(n_kept, len(singular_values), n_features - 1)
        return cls(mean=mean, components=right_vectors[:n_kept])

    def residuals(self, X: FloatArray) -> FloatArray:
        """
        Compute squared distances from the subspace.

        Parameters
        ----------
        X : FloatArray
            The samples with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The squared reconstruction errors with shape (n_samples,).
        """
        centered: FloatArray = X - self.mean
        # Subtracting the projection, rather than its norm, avoids cancellation.
        residual: FloatArray = (
            centered - (centered @ self.components.T) @ self.components
        )
        squared_errors: FloatArray = np.square(residual).sum(axis=1)
        return squared_errors
