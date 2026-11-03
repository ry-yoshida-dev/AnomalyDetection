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
| [methods/](./methods/README.md) | Per-method parameter classes. |
| [processors/](./processors/README.md) | Concrete processors, split by how they obtain training scores. |
