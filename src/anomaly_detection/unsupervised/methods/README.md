# methods

## Overview

Parameter classes for each unsupervised method. Every parameter class is frozen, validates its fields on construction, and builds an unfitted processor via `build_processor()`.

## Components

| Component | Description |
|-----------|-------------|
| [isolation_forest.py](./isolation_forest.py) | `IsolationForestParameters`. |
| [local_outlier_factor.py](./local_outlier_factor.py) | `LocalOutlierFactorParameters`. |
| [one_class_svm.py](./one_class_svm.py) | `OneClassSVMParameters`. |
| [mahalanobis.py](./mahalanobis.py) | `MahalanobisParameters`. |
| [k_nearest_neighbors.py](./k_nearest_neighbors.py) | `KNearestNeighborsParameters`. |
