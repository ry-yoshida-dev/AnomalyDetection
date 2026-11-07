# unsupervised

## Overview

Methods that learn normality from unlabeled, mostly normal samples. `fit` sets the threshold from the training scores, so `contamination` means the same for every method.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `UnsupervisedParameters`, the abstract base class holding `contamination`. |
| [processor.py](./processor.py) | `UnsupervisedProcessor`, the abstract base class implementing scaling, `fit`, and the threshold. |
| [svm_kernel.py](./svm_kernel.py) | `SVMKernel` enum of one-class SVM kernels. |
| [svm_gamma.py](./svm_gamma.py) | `SVMGamma` enum of kernel coefficient rules. |
| [sample_scorer.py](./sample_scorer.py) | `SampleScorerLike`, the structural type of estimators with `score_samples`. |
| [isolation_forest/](./isolation_forest/README.md) | Isolation forest parameters. |
| [one_class_svm/](./one_class_svm/README.md) | One-class SVM parameters. |
| [local_outlier_factor/](./local_outlier_factor/README.md) | Local outlier factor parameters and processor. |
| [k_nearest_neighbors/](./k_nearest_neighbors/README.md) | k-nearest neighbors parameters and processor. |
| [mahalanobis/](./mahalanobis/README.md) | Mahalanobis distance parameters and processor. |
| [sample_scoring/](./sample_scoring/README.md) | The processor shared by isolation forest and one-class SVM. |
