# one_class_svm

## Overview

One-class SVM, scored through `SampleScoringProcessor`, which scores the training samples out of fold. Scores are normalized so that models fitted on different numbers of samples share one scale.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `OneClassSVMParameters`, which validates the settings and builds the processor. |
| [kernel.py](./kernel.py) | `SVMKernel` enum of kernels. |
| [gamma.py](./gamma.py) | `SVMGamma` enum of kernel coefficient rules. |
| [normalized_estimator.py](./normalized_estimator.py) | `NormalizedOneClassSVM`, whose scores do not grow with the number of training samples. |
