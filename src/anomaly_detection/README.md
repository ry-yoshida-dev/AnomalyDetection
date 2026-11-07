# anomaly_detection

## Overview

Normal or anomalous decisions on feature vectors. Each family of methods lives in its own package, differing by what it needs before it can decide; all of them build an `AnomalyProcessor` with `score` and `predict`.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases for features, scores, and decisions. |
| [feature_validator.py](./feature_validator.py) | `FeatureValidator`, runtime checks of feature matrices, as strict as those of DataScaler. |
| [parameter_validator.py](./parameter_validator.py) | `ParameterValidator`, runtime checks of parameter types. |
| [neighbors_metric.py](./neighbors_metric.py) | `NeighborsMetric` enum of distance metrics. |
| [processor.py](./processor.py) | `AnomalyProcessor`, the abstract base class implementing `score` and `predict`. Families that scale features leave the feature checks to the scaler. |
| [parameter.py](./parameter.py) | `AnomalyParameters`, the abstract base class of parameters that build a processor. |
| [supervised/](./supervised/README.md) | Methods that learn from normal and anomalous samples. |
| [unsupervised/](./unsupervised/README.md) | Methods that learn from unlabeled, mostly normal samples. |
| [zero_shot/](./zero_shot/README.md) | Methods that need no training samples. |
