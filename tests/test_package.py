from __future__ import annotations

import re
import unittest
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
