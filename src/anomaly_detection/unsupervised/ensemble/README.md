# ensemble

## Overview

Combines several unsupervised methods by averaging the percentiles of their scores among their own out-of-fold training scores, so methods on different score scales weigh equally.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `EnsembleParameters`, which validates the members and builds the processor. |
| [processor.py](./processor.py) | `EnsembleProcessor`, which fits the members and averages their score percentiles. |

## Examples

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
    contamination=0.01,
).build_processor()
processor.fit(X_train)
is_anomalous = processor.predict(X_test)
```
