"""Type aliases for feature matrices, scores, and decisions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
