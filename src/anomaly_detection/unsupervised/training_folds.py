from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..types import BoolArray

# Each model is fitted on four fifths of the samples, close to the final model.
N_FOLDS: Final[int] = 5


class TrainingFolds:
    """
    Splits training samples so each can be scored by a model fitted without it.

    Scores of samples a model was fitted on are biased toward normal, so a
    threshold taken from them flags too many unseen samples.
    """

    @staticmethod
    def held_out_masks(n_samples: int) -> list[BoolArray]:
        """
        Assign the samples to folds in turn, so sorted inputs spread evenly.

        Parameters
        ----------
        n_samples : int
            The number of training samples, at least 2.

        Returns
        -------
        list[BoolArray]
            One mask per fold, True for the samples held out of it.

        Raises
        ------
        ValueError
            If there are fewer than 2 samples.
        """
        if n_samples < 2:
            raise ValueError(f"At least 2 samples are required, got {n_samples}")
        n_folds: int = min(N_FOLDS, n_samples)
        fold_indices: NDArray[np.intp] = np.arange(n_samples) % n_folds
        return [fold_indices == fold for fold in range(n_folds)]
