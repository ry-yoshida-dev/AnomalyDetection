from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from anomaly_detection import FeatureScaling, NeighborsMetric
from anomaly_detection.unsupervised import (
    IsolationForestParameters,
    KNearestNeighborsParameters,
    LocalOutlierFactorParameters,
    MahalanobisParameters,
    OneClassSVMParameters,
    UnsupervisedParameters,
    UnsupervisedProcessor,
)

from .data import SyntheticFeatures

ALL_METHODS: tuple[UnsupervisedParameters, ...] = (
    IsolationForestParameters(),
    LocalOutlierFactorParameters(),
    OneClassSVMParameters(),
    MahalanobisParameters(),
    KNearestNeighborsParameters(),
)


class TestAllMethods(unittest.TestCase):
    def test_detects_far_samples_and_ranks_them_higher(self) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal()
        anomalous: NDArray[np.float64] = SyntheticFeatures.anomalous()
        for parameters in ALL_METHODS:
            with self.subTest(method=type(parameters).__name__):
                processor: UnsupervisedProcessor = parameters.build_processor()
                processor.fit(X)
                self.assertTrue(processor.predict(anomalous).all())
                self.assertGreater(
                    processor.score(anomalous).min(), np.median(processor.score(X))
                )

    def test_false_positive_rate_on_unseen_normal_samples_follows_contamination(
        self,
    ) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal(n_samples=1000)
        unseen: NDArray[np.float64] = SyntheticFeatures.normal(n_samples=2000, seed=5)
        for parameters in (
            IsolationForestParameters(contamination=0.1),
            LocalOutlierFactorParameters(contamination=0.1),
            OneClassSVMParameters(contamination=0.1),
            MahalanobisParameters(contamination=0.1),
            KNearestNeighborsParameters(contamination=0.1),
        ):
            with self.subTest(method=type(parameters).__name__):
                processor: UnsupervisedProcessor = parameters.build_processor()
                processor.fit(X)
                self.assertAlmostEqual(
                    float(processor.predict(unseen).mean()), 0.1, delta=0.04
                )

    def test_requires_fit(self) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal()
        for parameters in ALL_METHODS:
            with self.subTest(method=type(parameters).__name__):
                processor: UnsupervisedProcessor = parameters.build_processor()
                with self.assertRaises(NotFittedError):
                    _ = processor.predict(X)
                with self.assertRaises(NotFittedError):
                    _ = processor.threshold

    def test_failed_fit_leaves_the_processor_unfitted(self) -> None:
        processor: UnsupervisedProcessor = KNearestNeighborsParameters(
            n_neighbors=5
        ).build_processor()
        processor.fit(SyntheticFeatures.normal())
        with self.assertRaises(ValueError):
            processor.fit(SyntheticFeatures.normal(n_samples=3))
        with self.assertRaises(NotFittedError):
            _ = processor.predict(SyntheticFeatures.normal())

    def test_rejects_invalid_features(self) -> None:
        processor: UnsupervisedProcessor = MahalanobisParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(np.array([[np.nan, 1.0], [0.0, 1.0]]))
        with self.assertRaises(TypeError):
            processor.fit(np.ma.masked_array(SyntheticFeatures.normal()))
        processor.fit(SyntheticFeatures.normal())
        with self.assertRaises(ValueError):
            _ = processor.score(np.array([[np.inf, 1.0]]))
        with self.assertRaises(TypeError):
            _ = processor.score(np.ma.masked_array(SyntheticFeatures.normal()))


class TestScaling(unittest.TestCase):
    def test_every_scaling_fits(self) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal()
        for scaling in FeatureScaling:
            with self.subTest(scaling=scaling):
                processor: UnsupervisedProcessor = KNearestNeighborsParameters(
                    scaling=scaling
                ).build_processor()
                processor.fit(X)
                self.assertEqual(processor.score(X).shape, (200,))

    def test_scaling_keeps_a_large_feature_from_dominating(self) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal()
        X[:, 0] *= 1e6
        anomalous: NDArray[np.float64] = np.array([[0.0, 12.0, 12.0, 12.0]])
        processor: UnsupervisedProcessor = (
            KNearestNeighborsParameters().build_processor()
        )
        processor.fit(X)
        self.assertTrue(processor.predict(anomalous)[0])


class TestKNearestNeighbors(unittest.TestCase):
    def test_training_scores_exclude_the_sample_itself(self) -> None:
        X: NDArray[np.float64] = np.array([[0.0], [1.0], [3.0]])
        processor: UnsupervisedProcessor = KNearestNeighborsParameters(
            n_neighbors=1, contamination=0.5, scaling=FeatureScaling.NONE
        ).build_processor()
        processor.fit(X)
        # Distances to the nearest other sample are 1, 1, and 2.
        self.assertEqual(processor.threshold, 1.0)

    def test_supports_cosine_distance(self) -> None:
        processor: UnsupervisedProcessor = KNearestNeighborsParameters(
            metric=NeighborsMetric.COSINE, scaling=FeatureScaling.NONE
        ).build_processor()
        X: NDArray[np.float64] = SyntheticFeatures.normal() * 0.01 + np.array(
            [1.0, 0.0, 0.0, 0.0]
        )
        processor.fit(X)
        self.assertTrue(processor.predict(np.array([[0.0, 5.0, 0.0, 0.0]]))[0])


class TestParameterValidation(unittest.TestCase):
    def test_rejects_contamination_outside_the_unit_interval(self) -> None:
        for contamination in (0.0, 1.0, float("nan")):
            with (
                self.subTest(contamination=contamination),
                self.assertRaises(ValueError),
            ):
                _ = MahalanobisParameters(contamination=contamination)

    def test_rejects_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            _ = IsolationForestParameters(n_estimators=0)
        with self.assertRaises(ValueError):
            _ = IsolationForestParameters(max_samples=0)
        with self.assertRaises(ValueError):
            _ = LocalOutlierFactorParameters(n_neighbors=0)
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(n_neighbors=0)
        with self.assertRaises(TypeError):
            _ = KNearestNeighborsParameters(n_neighbors=cast(int, 1.5))

    def test_rejects_invalid_one_class_svm_values(self) -> None:
        for nu in (0.0, 1.5, float("nan")):
            with self.subTest(nu=nu), self.assertRaises(ValueError):
                _ = OneClassSVMParameters(nu=nu)
        for gamma in (0.0, float("nan")):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                _ = OneClassSVMParameters(gamma=gamma)
