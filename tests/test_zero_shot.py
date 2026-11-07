from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from anomaly_detection import AnomalyProcessor, NeighborsMetric
from anomaly_detection.zero_shot import (
    MutualScoringParameters,
    PrototypeSimilarityParameters,
)

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
