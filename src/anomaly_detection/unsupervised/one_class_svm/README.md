# one_class_svm

## Overview

One-class SVM, scored through `SampleScoringProcessor`, with scores that share one scale across training set sizes.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `OneClassSVMParameters`, which validates the settings and builds the processor. |
| [kernel.py](./kernel.py) | `SVMKernel` enum of kernels. |
| [gamma.py](./gamma.py) | `SVMGamma` enum of kernel coefficient rules. |
| [normalized_estimator.py](./normalized_estimator.py) | `NormalizedOneClassSVM`, whose scores do not grow with the number of training samples. |
