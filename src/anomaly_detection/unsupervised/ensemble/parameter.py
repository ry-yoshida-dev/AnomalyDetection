from __future__ import annotations

from dataclasses import dataclass
from typing import cast, override

from data_scaler import FeatureScaling

from ..parameter import UnsupervisedParameters
from .processor import EnsembleProcessor


@dataclass(frozen=True)
class EnsembleParameters(UnsupervisedParameters):
    """
    Parameters for averaging the score percentiles of several methods.

    Combining methods that see different kinds of deviation, such as isolation
    forest, k-nearest neighbors, and Mahalanobis distance, spares choosing one
    of them. Fitting costs the sum of the members' fits.

    Attributes
    ----------
    members : tuple[UnsupervisedParameters, ...]
        The methods to combine, at least one. Their own contamination is
        ignored; the ensemble's contamination sets the threshold.

    Raises
    ------
    TypeError
        If members is not a tuple of UnsupervisedParameters.
    ValueError
        If members is empty.
    """

    members: tuple[UnsupervisedParameters, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        members: object = cast(object, self.members)
        if not isinstance(members, tuple):
            raise TypeError(
                f"members must be a tuple of UnsupervisedParameters, got {type(members).__name__}"
            )
        for member in cast(tuple[object, ...], members):
            if not isinstance(member, UnsupervisedParameters):
                raise TypeError(
                    f"members must be UnsupervisedParameters, got {member!r}"
                )
        if not self.members:
            raise ValueError("members must not be empty")

    @override
    def build_processor(self) -> EnsembleProcessor:
        """
        Build an unfitted ensemble processor.

        Returns
        -------
        EnsembleProcessor
            The processor.
        """
        # Each member scales the features its own way.
        return EnsembleProcessor(
            contamination=self.contamination,
            scaling=FeatureScaling.NONE,
            members=tuple(member.build_processor() for member in self.members),
        )
