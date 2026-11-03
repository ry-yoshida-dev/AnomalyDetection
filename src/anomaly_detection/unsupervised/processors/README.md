# processors

## Overview

Concrete `UnsupervisedProcessor` implementations. They differ by the estimator they wrap and by how they obtain training scores that are comparable with scores of unseen samples.

## Components

| Component | Description |
|-----------|-------------|
| [sample_scoring.py](./sample_scoring.py) | `SampleScoringProcessor`, for isolation forest and one-class SVM. |
| [local_outlier_factor.py](./local_outlier_factor.py) | `LocalOutlierFactorProcessor`, using the training factors that exclude each sample. |
| [k_nearest_neighbors.py](./k_nearest_neighbors.py) | `KNearestNeighborsProcessor`, using training distances that exclude each sample. |
| [mahalanobis.py](./mahalanobis.py) | `MahalanobisProcessor`, using a shrinkage covariance estimate. |
