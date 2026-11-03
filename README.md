# AnomalyDetection

## Overview

AnomalyDetection (`anomaly_detection`) decides whether feature vectors are normal or anomalous. Extracting the features is the caller's responsibility, as is measuring performance. Parameter dataclasses validate their fields and build a processor, and every processor offers `score` (higher is more anomalous) and `predict` (`True` is anomalous).

Choose the package by what is available before detection:

| Package | Available beforehand | Preparation | Methods |
| --- | --- | --- | --- |
| `anomaly_detection.supervised` | Normal and anomalous samples | `fit(X, y)` | Logistic regression, random forest, histogram gradient boosting, or any classifier with `fit` and `predict_proba` |
| `anomaly_detection.unsupervised` | Unlabeled, mostly normal samples | `fit(X)` | Isolation forest, local outlier factor, one-class SVM, Mahalanobis distance, k-nearest neighbors |
| `anomaly_detection.zero_shot` | No training samples | None | Prototype similarity, mutual scoring |

For module-level design notes, see [src/anomaly_detection/README.md](src/anomaly_detection/README.md).

## Installation

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
- Local outlier factor and k-nearest neighbors exclude each training sample from its own neighbors when setting the threshold, which keeps it valid for unseen samples. Scoring a training sample with `score` still counts it as its own neighbor, so such scores are lower than those of comparable unseen samples.
- Supervised scores are probabilities of the anomalous class. Classes are balanced by default, because anomalies are usually rare.
- Feature scaling comes from [DataScaler](https://github.com/ry-yoshida-dev/DataScaler), and `FeatureScaling` is re-exported here. Apart from it, the package depends only on NumPy and scikit-learn.
- Mutual scoring assumes anomalies are rare in the batch. Its scores and threshold are relative to the batch, so they are not comparable across batches, and batches must have more samples than `n_neighbors`.
- Prototypes must come from the same embedding space as the samples. Several prototypes of one kind are averaged by direction.
- NaN and infinite features are rejected with `ValueError`, also for classifiers that could handle them.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on every push to `main` and every pull request, along with the tests on the minimum supported dependency versions.

## License

Released under the [MIT License](LICENSE).
