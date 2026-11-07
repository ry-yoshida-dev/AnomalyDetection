from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray

from anomaly_detection import AnomalyProcessor, NeighborsMetric
from anomaly_detection.zero_shot import (
    MutualScoringParameters,
    PrototypeSimilarityParameters,
)
from anomaly_detection.zero_shot.unit_normalizer import UnitNormalizer

from .data import SyntheticFeatures

ORTHOGONAL_PROTOTYPES: PrototypeSimilarityParameters = PrototypeSimilarityParameters(
    normal_prototypes=np.array([[1.0, 0.0]]),
    anomalous_prototypes=np.array([[0.0, 1.0]]),
)
EUCLIDEAN_SCORING: MutualScoringParameters = MutualScoringParameters(
    metric=NeighborsMetric.EUCLIDEAN
)


class TestPrototypeSimilarity(unittest.TestCase):
    def test_decides_by_the_closer_prototype_without_fitting(self) -> None:
        X: NDArray[np.float64] = np.array([[2.0, 0.1], [0.1, 3.0]])
        decisions: NDArray[np.bool_] = ORTHOGONAL_PROTOTYPES.build_processor().predict(
            X
        )
        self.assertEqual(decisions.tolist(), [False, True])

    def test_scores_are_probabilities_even_at_extreme_logits(self) -> None:
        parameters: PrototypeSimilarityParameters = PrototypeSimilarityParameters(
            normal_prototypes=np.array([[1.0, 0.0]]),
            anomalous_prototypes=np.array([[0.0, 1.0]]),
            temperature=1e-6,
        )
        scores: NDArray[np.floating] = parameters.build_processor().score(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        self.assertEqual(scores.tolist(), [0.0, 1.0, 0.5])

    def test_merges_several_prototypes_by_direction(self) -> None:
        parameters: PrototypeSimilarityParameters = PrototypeSimilarityParameters(
            normal_prototypes=np.array([[100.0, 0.0], [0.0, 1.0]]),
            anomalous_prototypes=np.array([[-1.0, -1.0]]),
        )
        processor: AnomalyProcessor = parameters.build_processor()
        self.assertEqual(processor.predict(np.array([[1.0, 1.0]])).tolist(), [False])
        self.assertEqual(processor.predict(np.array([[-1.0, -1.0]])).tolist(), [True])

    def test_rejects_mismatched_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            _ = PrototypeSimilarityParameters(
                normal_prototypes=np.zeros((1, 2)) + 1.0,
                anomalous_prototypes=np.zeros((1, 3)) + 1.0,
            )
        with self.assertRaises(ValueError):
            _ = ORTHOGONAL_PROTOTYPES.build_processor().predict(np.ones((1, 3)))

    def test_rejects_vectors_without_direction(self) -> None:
        with self.assertRaises(ValueError):
            _ = PrototypeSimilarityParameters(
                normal_prototypes=np.zeros((1, 2)),
                anomalous_prototypes=np.ones((1, 2)),
            )
        with self.assertRaises(ValueError):
            _ = PrototypeSimilarityParameters(
                normal_prototypes=np.array([[1.0, 0.0], [-1.0, 0.0]]),
                anomalous_prototypes=np.ones((1, 2)),
            )
        with self.assertRaises(ValueError):
            _ = ORTHOGONAL_PROTOTYPES.build_processor().score(np.zeros((1, 2)))

    def test_rejects_invalid_values(self) -> None:
        normal: NDArray[np.float64] = np.array([[1.0, 0.0]])
        anomalous: NDArray[np.float64] = np.array([[0.0, 1.0]])
        for temperature in (0.0, float("nan")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                _ = PrototypeSimilarityParameters(
                    normal, anomalous, temperature=temperature
                )
        for threshold in (0.0, 1.0, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                _ = PrototypeSimilarityParameters(
                    normal, anomalous, decision_threshold=threshold
                )


class TestMutualScoring(unittest.TestCase):
    def test_flags_samples_unlike_the_batch(self) -> None:
        X: NDArray[np.float64] = np.vstack(
            [SyntheticFeatures.normal(), SyntheticFeatures.anomalous(n_samples=3)]
        )
        decisions: NDArray[np.bool_] = EUCLIDEAN_SCORING.build_processor().predict(X)
        self.assertTrue(decisions[-3:].all())
        self.assertLess(float(decisions[:-3].mean()), 0.05)

    def test_identical_samples_are_all_normal(self) -> None:
        decisions: NDArray[np.bool_] = EUCLIDEAN_SCORING.build_processor().predict(
            np.ones((10, 4))
        )
        self.assertFalse(decisions.any())

    def test_rejects_batches_smaller_than_the_neighborhood(self) -> None:
        with self.assertRaises(ValueError):
            _ = EUCLIDEAN_SCORING.build_processor().score(np.ones((5, 4)))

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            _ = MutualScoringParameters(n_neighbors=0)
        for threshold in (0.0, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                _ = MutualScoringParameters(deviation_threshold=threshold)


class TestFeatureChecks(unittest.TestCase):
    def test_rejects_masked_features_like_the_scaled_families(self) -> None:
        # Masking hides the NaN, which would otherwise reach the scores.
        X: NDArray[np.float64] = SyntheticFeatures.normal()
        X[0, 0] = np.nan
        cases: tuple[tuple[AnomalyProcessor, NDArray[np.float64]], ...] = (
            (ORTHOGONAL_PROTOTYPES.build_processor(), np.ma.masked_invalid(X[:, :2])),
            (EUCLIDEAN_SCORING.build_processor(), np.ma.masked_invalid(X)),
        )
        for processor, masked in cases:
            with (
                self.subTest(processor=type(processor).__name__),
                self.assertRaises(TypeError),
            ):
                _ = processor.score(masked)


class TestUnitNormalizer(unittest.TestCase):
    def test_integer_vectors_do_not_overflow(self) -> None:
        for X in (
            np.array([[100, 0]], dtype=np.int8),
            np.array([[50_000, 40_000]], dtype=np.int32),
            np.array([[2**40, 0]], dtype=np.int64),
        ):
            with self.subTest(dtype=X.dtype):
                unit: NDArray[np.floating] = UnitNormalizer.normalize(X)
                np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0)

    def test_extreme_floats_keep_their_direction(self) -> None:
        for X in (
            np.array([[1e200, 1e200]]),
            np.array([[1e-200, 1e-200]]),
            np.array([[1e-25, 1e-25]], dtype=np.float32),
        ):
            with self.subTest(values=X.tolist()):
                np.testing.assert_allclose(
                    UnitNormalizer.normalize(X), [[2**-0.5, 2**-0.5]]
                )

    def test_integer_samples_score_like_float_samples(self) -> None:
        X: NDArray[np.int32] = np.array(
            [[50_000, 40_000], [40_000, 50_000]], dtype=np.int32
        )
        processor: AnomalyProcessor = ORTHOGONAL_PROTOTYPES.build_processor()
        np.testing.assert_allclose(
            processor.score(X), processor.score(X.astype(np.float64))
        )


class TestPrototypeDirections(unittest.TestCase):
    def test_rejects_prototypes_that_nearly_cancel_out(self) -> None:
        with self.assertRaises(ValueError):
            _ = PrototypeSimilarityParameters(
                normal_prototypes=np.array([[1.0, 0.0], [-1.0, 1e-9]]),
                anomalous_prototypes=np.array([[0.0, 1.0]]),
            )

    def test_rejects_prototypes_that_do_not_separate_the_kinds(self) -> None:
        with self.assertRaises(ValueError):
            _ = PrototypeSimilarityParameters(
                normal_prototypes=np.array([[1.0, 2.0]]),
                anomalous_prototypes=np.array([[2.0, 4.0]]),
            )

    def test_rejects_an_infinite_temperature(self) -> None:
        with self.assertRaises(ValueError):
            _ = PrototypeSimilarityParameters(
                normal_prototypes=np.array([[1.0, 0.0]]),
                anomalous_prototypes=np.array([[0.0, 1.0]]),
                temperature=float("inf"),
            )


class TestMutualScoringSpread(unittest.TestCase):
    def test_near_copies_are_normal_when_most_samples_are_identical(self) -> None:
        X: NDArray[np.float64] = np.vstack(
            [np.ones((7, 2)), np.ones((5, 2)) + 1e-6 * np.arange(1, 6)[:, None]]
        )
        self.assertFalse(EUCLIDEAN_SCORING.build_processor().predict(X).any())

    def test_flags_a_far_sample_among_identical_ones(self) -> None:
        X: NDArray[np.float64] = np.vstack([np.ones((99, 2)), [[10.0, 10.0]]])
        decisions: NDArray[np.bool_] = EUCLIDEAN_SCORING.build_processor().predict(X)
        self.assertTrue(decisions[-1])
        self.assertFalse(decisions[:-1].any())

    def test_cosine_distance_flags_a_sample_pointing_elsewhere(self) -> None:
        X: NDArray[np.float64] = np.vstack(
            [
                SyntheticFeatures.normal() * 0.01 + [1.0, 0.0, 0.0, 0.0],
                [[0.0, 1.0, 0.0, 0.0]],
            ]
        )
        decisions: NDArray[np.bool_] = (
            MutualScoringParameters().build_processor().predict(X)
        )
        self.assertTrue(decisions[-1])

    def test_cosine_distance_rejects_vectors_without_direction(self) -> None:
        X: NDArray[np.float64] = SyntheticFeatures.normal()
        X[0] = 0.0
        with self.assertRaises(ValueError):
            _ = MutualScoringParameters().build_processor().score(X)

    def test_rejects_values_that_would_fail_or_never_flag(self) -> None:
        with self.assertRaises(ValueError):
            _ = MutualScoringParameters(n_jobs=0)
        with self.assertRaises(ValueError):
            _ = MutualScoringParameters(deviation_threshold=float("inf"))
        with self.assertRaises(TypeError):
            _ = MutualScoringParameters(
                metric=cast(NeighborsMetric, cast(object, "cosine"))
            )
