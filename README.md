# AnomalyDetection

## Overview

AnomalyDetection (`anomaly_detection`) decides whether feature vectors are normal or anomalous. Parameter dataclasses validate their fields and build a processor, and every processor offers `score` (higher is more anomalous) and `predict` (`True` is anomalous). Choose the package by what is available before detection:

| Package | Available beforehand | Methods |
| --- | --- | --- |
| `anomaly_detection.supervised` | Normal and anomalous samples | Logistic regression, random forest, histogram gradient boosting, or any classifier with `fit` and `predict_proba` |
| `anomaly_detection.unsupervised` | Unlabeled, mostly normal samples | Isolation forest, local outlier factor, one-class SVM, Mahalanobis distance, k-nearest neighbors, principal component reconstruction error, ECOD, Gaussian mixture, and an ensemble of them |
| `anomaly_detection.zero_shot` | No training samples | Prototype similarity, mutual scoring |

## Installation

Requires Python 3.12 or later.

```bash
pip install .
```

## Usage

Learn normality from unlabeled samples:

```python
from anomaly_detection.unsupervised import IsolationForestParameters

processor = IsolationForestParameters(contamination=0.01).build_processor()
processor.fit(X_train)
is_anomalous = processor.predict(X_test)
```

Learn from labeled samples, where `True` means anomalous:

```python
from anomaly_detection.supervised import RandomForestParameters

processor = RandomForestParameters().build_processor()
processor.fit(X_train, y_train)
is_anomalous = processor.predict(X_test)
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

## License

Released under the [MIT License](LICENSE).
