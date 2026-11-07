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
    PrincipalComponentParameters,
    SVMGamma,
    SVMKernel,
    UnsupervisedParameters,
    UnsupervisedProcessor,
)

from .data import SyntheticFeatures

# Methods that flag samples far from isotropic training samples in every direction.
DISTANCE_METHODS: tuple[UnsupervisedParameters, ...] = (
    IsolationForestParameters(),
    LocalOutlierFactorParameters(),
    OneClassSVMParameters(),
    MahalanobisParameters(),
    KNearestNeighborsParameters(),
)
ALL_METHODS: tuple[UnsupervisedParameters, ...] = (
    *DISTANCE_METHODS,
    PrincipalComponentParameters(),
)


class TestAllMethods(unittest.TestCase):
    def test_detects_far_samples_and_ranks_them_higher(self) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal()
        anomalous: NDArray[np.float64] = SyntheticFeatures.anomalous()
        for parameters in DISTANCE_METHODS:
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
        for parameters in ALL_METHODS:
            parameters = type(parameters)(contamination=0.1)
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


class TestOutOfFoldThreshold(unittest.TestCase):
    def test_false_positive_rate_holds_for_small_training_sets(self) -> None:
        # Methods scoring their own training samples would flag far more here.
        for parameters in ALL_METHODS:
            parameters = type(parameters)(contamination=0.1)
            with self.subTest(method=type(parameters).__name__):
                rates: list[float] = []
                for seed in range(5):
                    processor: UnsupervisedProcessor = parameters.build_processor()
                    processor.fit(SyntheticFeatures.normal(n_samples=100, seed=seed))
                    unseen: NDArray[np.float64] = SyntheticFeatures.normal(
                        n_samples=2000, seed=100 + seed
                    )
                    rates.append(float(processor.predict(unseen).mean()))
                self.assertAlmostEqual(float(np.mean(rates)), 0.1, delta=0.05)

    def test_rejects_identical_training_samples(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(method=type(parameters).__name__):
                processor: UnsupervisedProcessor = parameters.build_processor()
                with self.assertRaises(ValueError):
                    processor.fit(np.ones((50, 4)))
                with self.assertRaises(ValueError):
                    processor.fit(np.ones((1, 4)))


class TestCosineScaling(unittest.TestCase):
    def test_rejects_scalings_that_move_the_origin(self) -> None:
        for parameters_type in (
            KNearestNeighborsParameters,
            LocalOutlierFactorParameters,
        ):
            for scaling in (
                FeatureScaling.STANDARD,
                FeatureScaling.MIN_MAX,
                FeatureScaling.ROBUST,
            ):
                with (
                    self.subTest(method=parameters_type.__name__, scaling=scaling),
                    self.assertRaises(ValueError),
                ):
                    _ = parameters_type(metric=NeighborsMetric.COSINE, scaling=scaling)

    def test_accepts_scalings_that_keep_the_origin(self) -> None:
        for scaling in (
            FeatureScaling.NONE,
            FeatureScaling.MAX_ABS,
            FeatureScaling.UNIT_NORM,
        ):
            with self.subTest(scaling=scaling):
                _ = KNearestNeighborsParameters(
                    metric=NeighborsMetric.COSINE, scaling=scaling
                )


class TestPrincipalComponent(unittest.TestCase):
    def test_flags_samples_off_the_subspace_of_normal_samples(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        mixing: NDArray[np.float64] = rng.standard_normal((3, 20))
        X: NDArray[np.float64] = rng.standard_normal(
            (500, 3)
        ) @ mixing + 0.05 * rng.standard_normal((500, 20))
        on_subspace: NDArray[np.float64] = rng.standard_normal((50, 3)) @ mixing
        off_subspace: NDArray[np.float64] = on_subspace + rng.standard_normal((50, 20))
        processor: UnsupervisedProcessor = PrincipalComponentParameters(
            n_components=3
        ).build_processor()
        processor.fit(X)
        self.assertTrue(processor.predict(off_subspace).all())
        self.assertLess(float(processor.predict(on_subspace).mean()), 0.1)

    def test_explained_variance_keeps_the_dominant_directions(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        X: NDArray[np.float64] = rng.standard_normal((500, 4)) * np.array(
            [10.0, 10.0, 0.1, 0.1]
        )
        processor: UnsupervisedProcessor = PrincipalComponentParameters(
            explained_variance=0.9, scaling=FeatureScaling.NONE
        ).build_processor()
        processor.fit(X)
        self.assertFalse(processor.predict(np.array([[30.0, -30.0, 0.0, 0.0]]))[0])
        self.assertTrue(processor.predict(np.array([[0.0, 0.0, 1.0, 1.0]]))[0])

    def test_rejects_too_many_components_or_a_single_feature(self) -> None:
        with self.assertRaises(ValueError):
            PrincipalComponentParameters(n_components=4).build_processor().fit(
                SyntheticFeatures.normal()
            )
        with self.assertRaises(ValueError):
            PrincipalComponentParameters().build_processor().fit(
                SyntheticFeatures.normal()[:, :1]
            )

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            _ = PrincipalComponentParameters(n_components=0)
        for explained_variance in (0.0, 1.0, float("nan")):
            with (
                self.subTest(explained_variance=explained_variance),
                self.assertRaises(ValueError),
            ):
                _ = PrincipalComponentParameters(explained_variance=explained_variance)
        with self.assertRaises(TypeError):
            _ = PrincipalComponentParameters(n_components=cast(int, 2.0))


class TestOneClassSVM(unittest.TestCase):
    def test_every_kernel_and_gamma_rule_fits(self) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal()
        for kernel in SVMKernel:
            for gamma in SVMGamma:
                with self.subTest(kernel=kernel, gamma=gamma):
                    processor: UnsupervisedProcessor = OneClassSVMParameters(
                        kernel=kernel, gamma=gamma
                    ).build_processor()
                    processor.fit(X)
                    self.assertTrue(np.isfinite(processor.score(X)).all())

    def test_scores_do_not_depend_on_the_number_of_training_samples(self) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal(n_samples=1000)
        small: UnsupervisedProcessor = OneClassSVMParameters(
            scaling=FeatureScaling.NONE
        ).build_processor()
        large: UnsupervisedProcessor = OneClassSVMParameters(
            scaling=FeatureScaling.NONE
        ).build_processor()
        small.fit(X[:500])
        large.fit(X)
        probe: NDArray[np.float64] = np.zeros((1, 4))
        np.testing.assert_allclose(small.score(probe), large.score(probe), atol=0.05)


class TestParameterTypesAndRanges(unittest.TestCase):
    def test_rejects_values_that_would_fail_at_fit(self) -> None:
        with self.assertRaises(ValueError):
            _ = IsolationForestParameters(n_jobs=0)
        with self.assertRaises(ValueError):
            _ = IsolationForestParameters(random_state=-1)
        with self.assertRaises(ValueError):
            _ = IsolationForestParameters(random_state=2**32)
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(n_jobs=0)
        with self.assertRaises(ValueError):
            _ = LocalOutlierFactorParameters(n_jobs=0)
        with self.assertRaises(ValueError):
            _ = OneClassSVMParameters(gamma=float("inf"))

    def test_rejects_values_of_the_wrong_type(self) -> None:
        with self.assertRaises(TypeError):
            _ = OneClassSVMParameters(gamma=cast(float, True))
        with self.assertRaises(TypeError):
            _ = OneClassSVMParameters(kernel=cast(SVMKernel, cast(object, "rbf")))
        with self.assertRaises(TypeError):
            _ = KNearestNeighborsParameters(
                metric=cast(NeighborsMetric, cast(object, "cosine"))
            )
        with self.assertRaises(TypeError):
            _ = MahalanobisParameters(
                scaling=cast(FeatureScaling, cast(object, "standard"))
            )
        with self.assertRaises(TypeError):
            _ = MahalanobisParameters(contamination=cast(float, cast(object, "0.1")))
