# AnomalyDetection

## Overview

AnomalyDetection (`anomaly_detection`) decides whether feature vectors are normal or anomalous. Extracting the features is the caller's responsibility, as is measuring performance. Parameter dataclasses validate their fields and build a processor, and every processor offers `score` (higher is more anomalous) and `predict` (`True` is anomalous).

Choose the package by what is available before detection:

| Package | Available beforehand | Preparation | Methods |
| --- | --- | --- | --- |
| `anomaly_detection.supervised` | Normal and anomalous samples | `fit(X, y)` | Logistic regression, random forest, histogram gradient boosting, or any classifier with `fit` and `predict_proba` |
| `anomaly_detection.unsupervised` | Unlabeled, mostly normal samples | `fit(X)` | Isolation forest, local outlier factor, one-class SVM, Mahalanobis distance, k-nearest neighbors, principal component reconstruction error, ECOD, Gaussian mixture, and an ensemble of them |
| `anomaly_detection.zero_shot` | No training samples | None | Prototype similarity, mutual scoring |

For module-level design notes, see [src/anomaly_detection/README.md](src/anomaly_detection/README.md).

## Installation

Requires Python 3.12 or later; CI tests 3.12, 3.13, 3.14, and 3.15.

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Quick example

Learn normality from unlabeled samples:

```python
from anomaly_detection.unsupervised import IsolationForestParameters

processor = IsolationForestParameters(contamination=0.01).build_processor()
processor.fit(X_train)
is_anomalous = processor.predict(X_test)
scores = processor.score(X_test)
```

Combine several methods, so that none has to be chosen:

```python
from anomaly_detection.unsupervised import (
    EnsembleParameters,
    IsolationForestParameters,
    KNearestNeighborsParameters,
    MahalanobisParameters,
)

processor = EnsembleParameters(
    members=(
        IsolationForestParameters(),
        KNearestNeighborsParameters(),
        MahalanobisParameters(),
    ),
).build_processor()
processor.fit(X_train)
```

Learn from labeled samples, where `True` means anomalous:

```python
from anomaly_detection.supervised import RandomForestParameters

processor = RandomForestParameters().build_processor()
processor.fit(X_train, y_train)
is_anomalous = processor.predict(X_test)
```

Use another classifier by passing it directly:

```python
from anomaly_detection import FeatureScaling
from anomaly_detection.supervised import SupervisedProcessor

processor = SupervisedProcessor(
    classifier=my_classifier,
    scaling=FeatureScaling.STANDARD,
    decision_threshold=0.5,
)
```

Decide without training, from embeddings of what normal and anomalous look like:

```python
from anomaly_detection.zero_shot import PrototypeSimilarityParameters

processor = PrototypeSimilarityParameters(
    normal_prototypes=normal_text_embeddings,
    anomalous_prototypes=anomalous_text_embeddings,
).build_processor()
is_anomalous = processor.predict(image_embeddings)
```

Or from the batch itself, when no reference exists at all:

```python
from anomaly_detection.zero_shot import MutualScoringParameters

is_anomalous = MutualScoringParameters().build_processor().predict(X_batch)
```

## Notes

- A sample is anomalous when its score is strictly above the threshold.
- Unsupervised thresholds are the quantile of the training scores that leaves `contamination` above it, for every method. With purely normal training samples, `contamination` approximates the false positive rate on unseen samples.
- The threshold is set from training scores that each leave out the sample's own influence, which keeps it valid for unseen samples. Local outlier factor and k-nearest neighbors exclude each sample from its own neighbors; isolation forest, one-class SVM, Mahalanobis distance, Gaussian mixture, and principal component reconstruction error score each sample with a model fitted on the other four fifths of the samples (5-fold cross-fitting), so their `fit` costs about six fits. Scoring a training sample with `score` still uses the model fitted on all samples, so such scores are lower than those of comparable unseen samples.
- ECOD needs no hyperparameters, so it is a good first method to try. It treats features independently, so it misses anomalies that only break correlations between features, and its tail probabilities leave each training sample out instead of cross-fitting.
- Principal component reconstruction error flags only deviations outside the directions along which training samples vary, which suits embeddings; a sample far away along those directions stays normal. It always leaves at least one direction out, so features need at least two dimensions.
- An ensemble averages each member's percentile among its own out-of-fold training scores, so members on different score scales weigh equally. Percentiles stop at 1 beyond every training score, so the ensemble ranks such samples as ties.
- Unsupervised `fit` rejects training samples that are all identical, because they carry no spread to learn from.
- Cosine distance in k-nearest neighbors and local outlier factor needs a scaling that keeps the origin (`NONE`, `MAX_ABS`, or `UNIT_NORM`), because centering turns typical samples into zero vectors, which have no direction.
- Supervised scores are probabilities of the anomalous class. Classes are balanced by default, because anomalies are usually rare. Histogram gradient boosting trains without early stopping, whose validation split would hold back some of the few anomalies.
- Parameter classes check the types and ranges of their fields on construction, including values scikit-learn would only reject at `fit`, such as `n_jobs=0` or a seed outside [0, 2**32 - 1].
- Feature scaling comes from [DataScaler](https://github.com/ry-yoshida-dev/DataScaler), and `FeatureScaling` is re-exported here. Apart from it, the package depends only on NumPy and scikit-learn.
- Mutual scoring assumes anomalies are rare in the batch. Its threshold is the median score plus `deviation_threshold` scaled median absolute deviations, falling back to the mean absolute deviation when over half of the scores are equal. Its scores and threshold are relative to the batch, so they are not comparable across batches, and batches must have more samples than `n_neighbors`.
- Prototypes must come from the same embedding space as the samples. Several prototypes of one kind are averaged by direction.
- Features must be a plain numeric NumPy array, the same for every family. Masked arrays and `np.matrix` are rejected with `TypeError`, because their semantics would be lost; convert other inputs with `np.asarray`.
- NaN and infinite features are rejected with `ValueError`, also for classifiers that could handle them. Supervised and unsupervised processors also raise `ValueError` when scaled features cannot be represented, such as by overflowing.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on every push to `main` and every pull request, along with the tests on the minimum supported dependency versions for each Python version. Dependency resolution skips releases newer than 7 days (`exclude-newer` in `pyproject.toml`).

## License

Released under the [MIT License](LICENSE).
