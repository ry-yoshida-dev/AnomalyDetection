from __future__ import annotations

from typing import override

from numpy.typing import ArrayLike
from sklearn.svm import OneClassSVM

from ...types import FloatArray


class NormalizedOneClassSVM(OneClassSVM):
    """
    One-class SVM whose scores do not grow with the number of training samples.

    scikit-learn's scores sum kernel terms whose weights add up to nu times the
    number of training samples, so models fitted on different numbers of
    samples score on different scales. Dividing by the total weight makes the
    out-of-fold training scores comparable with those of the final model.
    """

    @override
    def score_samples(self, X: ArrayLike) -> FloatArray:
        """
        Compute normality scores as a weighted mean of the kernel terms.

        Parameters
        ----------
        X : ArrayLike
            The feature matrix.

        Returns
        -------
        FloatArray
            The scores with shape (n_samples,); higher is more normal.
        """
        return super().score_samples(X) / self.dual_coef_.sum()
