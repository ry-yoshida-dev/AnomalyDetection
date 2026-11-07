from __future__ import annotations

import re
import unittest
import warnings
from importlib.metadata import requires
from typing import cast

import numpy as np

import anomaly_detection
from anomaly_detection.feature_validator import FeatureValidator
from anomaly_detection.types import NumericArray


class TestPackageLayout(unittest.TestCase):
    def test_depends_only_on_the_intended_packages(self) -> None:
        requirements: list[str] = requires("anomaly-detection") or []
        names: set[str] = {
            re.split(r"[\s<>=!~;\[@]", requirement, maxsplit=1)[0]
            for requirement in requirements
        }
        self.assertEqual(names, {"data-scaler", "numpy", "scikit-learn"})

    def test_exports_shared_options(self) -> None:
        self.assertEqual(anomaly_detection.FeatureScaling.NONE.value, "none")
        self.assertEqual(anomaly_detection.NeighborsMetric.COSINE.value, "cosine")


class TestFeatureValidator(unittest.TestCase):
    def test_accepts_integer_and_float_matrices(self) -> None:
        FeatureValidator.require_matrix(np.zeros((2, 3), dtype=np.int64))
        FeatureValidator.require_matrix(np.zeros((2, 3), dtype=np.float32))

    def test_rejects_non_numeric_features(self) -> None:
        with self.assertRaises(TypeError):
            FeatureValidator.require_matrix(
                cast(NumericArray, np.zeros((2, 3), dtype=np.bool_))
            )

    def test_rejects_invalid_shapes_and_values(self) -> None:
        for X in (np.zeros(3), np.zeros((0, 3)), np.array([[np.nan, 1.0]])):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                FeatureValidator.require_matrix(X)

    def test_rejects_arrays_whose_semantics_would_be_lost(self) -> None:
        masked: NumericArray = np.ma.masked_invalid(
            np.array([[np.nan, 1.0], [2.0, 3.0]])
        )
        with warnings.catch_warnings():
            # NumPy discourages matrices, which is why callers may still pass one.
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            matrix: NumericArray = cast(NumericArray, np.asmatrix(np.zeros((2, 3))))
        not_an_array: NumericArray = cast(NumericArray, cast(object, [[0.0, 1.0]]))
        for X in (masked, matrix, not_an_array):
            with self.subTest(type=type(X).__name__), self.assertRaises(TypeError):
                FeatureValidator.require_matrix(X)
