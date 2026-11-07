from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from anomaly_detection import FeatureScaling
from anomaly_detection.supervised import (
    HistGradientBoostingParameters,
    LogisticRegressionParameters,
    RandomForestParameters,
    SupervisedParameters,
    SupervisedProcessor,
)
from anomaly_detection.types import BoolArray, FloatArray

from .data import SyntheticFeatures

X: NDArray[np.float64] = np.vstack(
    [SyntheticFeatures.normal(), SyntheticFeatures.anomalous(n_samples=30)]
)
Y: NDArray[np.bool_] = np.array([False] * 200 + [True] * 30)
ALL_METHODS: tuple[SupervisedParameters, ...] = (
    LogisticRegressionParameters(),
    RandomForestParameters(),
    HistGradientBoostingParameters(),
)


class TestAllMethods(unittest.TestCase):
    def test_separates_normal_and_anomalous_samples(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(method=type(parameters).__name__):
                processor: SupervisedProcessor = parameters.build_processor()
                processor.fit(X, Y)
                self.assertFalse(
                    processor.predict(SyntheticFeatures.normal(seed=2)).any()
                )
                self.assertTrue(
                    processor.predict(SyntheticFeatures.anomalous(seed=3)).all()
                )

    def test_requires_fit(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(method=type(parameters).__name__):
                with self.assertRaises(NotFittedError):
                    _ = parameters.build_processor().predict(X)


class TestSupervisedProcessor(unittest.TestCase):
    def test_score_is_the_anomalous_class_probability(self) -> None:
        processor: SupervisedProcessor = (
            LogisticRegressionParameters().build_processor()
        )
        processor.fit(X, Y)
        scores_normal: FloatArray = processor.score(SyntheticFeatures.normal(seed=2))
        scores_anomalous: FloatArray = processor.score(
            SyntheticFeatures.anomalous(seed=3)
        )
        self.assertLess(float(scores_normal.max()), 0.5)
        self.assertGreater(float(scores_anomalous.min()), 0.5)

    def test_threshold_changes_decisions(self) -> None:
        probe: NDArray[np.float64] = np.full((1, 4), 6.0)
        lenient: SupervisedProcessor = LogisticRegressionParameters(
            decision_threshold=0.001
        ).build_processor()
        strict: SupervisedProcessor = LogisticRegressionParameters(
            decision_threshold=0.999999
        ).build_processor()
        lenient.fit(X, Y)
        strict.fit(X, Y)
        self.assertTrue(lenient.predict(probe).item())
        self.assertFalse(strict.predict(probe).item())

    def test_accepts_any_structurally_matching_classifier(self) -> None:
        processor: SupervisedProcessor = SupervisedProcessor(
            classifier=RandomForestClassifier(n_estimators=10, random_state=0),
            scaling=FeatureScaling.ROBUST,
            decision_threshold=0.5,
        )
        processor.fit(X, Y)
        self.assertTrue(processor.predict(SyntheticFeatures.anomalous(seed=3)).all())

    def test_rejects_non_boolean_labels(self) -> None:
        processor: SupervisedProcessor = RandomForestParameters().build_processor()
        with self.assertRaises(TypeError):
            processor.fit(X, cast(BoolArray, Y.astype(np.int64)))

    def test_rejects_mismatched_labels(self) -> None:
        processor: SupervisedProcessor = RandomForestParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(X, Y[:-1])

    def test_rejects_non_vector_labels(self) -> None:
        processor: SupervisedProcessor = RandomForestParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(X, Y.reshape(-1, 1))

    def test_rejects_invalid_features(self) -> None:
        processor: SupervisedProcessor = RandomForestParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(np.full_like(X, np.nan), Y)
        with self.assertRaises(TypeError):
            processor.fit(np.ma.masked_array(X), Y)
        processor.fit(X, Y)
        with self.assertRaises(ValueError):
            _ = processor.score(np.full_like(X, np.inf))
        with self.assertRaises(TypeError):
            _ = processor.score(np.ma.masked_array(X))

    def test_rejects_a_single_class(self) -> None:
        processor: SupervisedProcessor = RandomForestParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(X, np.zeros(len(X), dtype=np.bool_))
        with self.assertRaises(ValueError):
            processor.fit(X, np.ones(len(X), dtype=np.bool_))

    def test_failed_fit_leaves_the_processor_unfitted(self) -> None:
        processor: SupervisedProcessor = RandomForestParameters().build_processor()
        processor.fit(X, Y)
        with self.assertRaises(ValueError):
            processor.fit(X, Y[:-1])
        with self.assertRaises(NotFittedError):
            _ = processor.predict(X)


class TestParameterValidation(unittest.TestCase):
    def test_rejects_thresholds_outside_the_unit_interval(self) -> None:
        for threshold in (0.0, 1.0, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                _ = RandomForestParameters(decision_threshold=threshold)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(C=0.0)
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(max_iter=0)
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(n_estimators=0)
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(max_depth=0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(learning_rate=float("nan"))
        with self.assertRaises(TypeError):
            _ = HistGradientBoostingParameters(max_iter=cast(int, 1.5))

    def test_rejects_values_that_would_fail_at_fit(self) -> None:
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(n_jobs=0)
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(random_state=-1)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(random_state=2**40)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(learning_rate=float("inf"))
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(min_samples_leaf=0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(max_depth=0)

    def test_rejects_values_of_the_wrong_type(self) -> None:
        with self.assertRaises(TypeError):
            _ = RandomForestParameters(is_class_balanced=cast(bool, cast(object, "no")))
        with self.assertRaises(TypeError):
            _ = LogisticRegressionParameters(
                scaling=cast(FeatureScaling, cast(object, "standard"))
            )
        with self.assertRaises(TypeError):
            _ = LogisticRegressionParameters(C=cast(float, cast(object, "1.0")))
        with self.assertRaises(TypeError):
            _ = RandomForestParameters(decision_threshold=cast(float, True))


class TestLabels(unittest.TestCase):
    def test_rejects_labels_that_are_not_plain_arrays(self) -> None:
        processor: SupervisedProcessor = RandomForestParameters().build_processor()
        with self.assertRaises(TypeError):
            processor.fit(X, cast(BoolArray, Y.tolist()))
        with self.assertRaises(TypeError):
            processor.fit(X, np.ma.masked_array(Y))


class TestHistGradientBoosting(unittest.TestCase):
    def test_trains_on_every_sample_beyond_the_early_stopping_size(self) -> None:
        # Early stopping would hold out a stratified split, which a single anomaly cannot fill.
        features: NDArray[np.float64] = SyntheticFeatures.normal(n_samples=10_001)
        labels: NDArray[np.bool_] = np.zeros(10_001, dtype=np.bool_)
        features[0] += 20.0
        labels[0] = True
        processor: SupervisedProcessor = HistGradientBoostingParameters(
            max_iter=10
        ).build_processor()
        processor.fit(features, labels)
        self.assertEqual(processor.score(features).shape, (10_001,))
