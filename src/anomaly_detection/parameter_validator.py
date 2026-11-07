from __future__ import annotations

from numbers import Integral, Real
from typing import Final

# scikit-learn seeds NumPy's legacy generator, which accepts 32-bit seeds only.
MAX_RANDOM_STATE: Final[int] = 2**32 - 1


class ParameterValidator:
    """
    Runtime checks that type annotations and range comparisons cannot express.
    """

    @staticmethod
    def require_integers(**values: object) -> None:
        """
        Check that values are integers, or None for optional parameters.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is neither None nor an integer; booleans are rejected.
        """
        for name, value in values.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {value!r}")

    @staticmethod
    def require_real_numbers(**values: object) -> None:
        """
        Check that values are real numbers.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not a real number; booleans are rejected.
        """
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, got {value!r}")

    @staticmethod
    def require_instances(expected: type, **values: object) -> None:
        """
        Check that values are instances of a type, such as an Enum or bool.

        Parameters
        ----------
        expected : type
            The required type.
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not an instance of the type.
        """
        for name, value in values.items():
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be a {expected.__name__}, got {value!r}")

    @staticmethod
    def require_job_count(n_jobs: int | None) -> None:
        """
        Check that a number of parallel jobs is meaningful.

        Parameters
        ----------
        n_jobs : int | None
            The number of jobs; negative values count back from all CPUs.

        Raises
        ------
        ValueError
            If n_jobs is 0.
        """
        if n_jobs == 0:
            raise ValueError(
                "n_jobs must not be 0; use None for one job or -1 for all CPUs"
            )

    @staticmethod
    def require_random_state(random_state: int | None) -> None:
        """
        Check that a seed is in the range scikit-learn accepts.

        Parameters
        ----------
        random_state : int | None
            The seed, or None for fresh randomness.

        Raises
        ------
        ValueError
            If the seed is outside [0, 2**32 - 1].
        """
        if random_state is not None and not 0 <= random_state <= MAX_RANDOM_STATE:
            raise ValueError(
                f"random_state must be in [0, {MAX_RANDOM_STATE}], got {random_state}"
            )
