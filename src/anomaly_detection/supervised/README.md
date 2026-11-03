# supervised

## Overview

Anomaly detection as binary classification, for when anomalous samples are available. Scores are the estimated probability of the anomalous class. Other classifiers can be used by passing any object matching `ClassifierLike` to `SupervisedProcessor`.

## Components

| Component | Description |
|-----------|-------------|
| [classifier.py](./classifier.py) | `ClassifierLike`, the structural type of wrapped classifiers. |
| [parameter.py](./parameter.py) | `SupervisedParameters`, the abstract base class holding the threshold and class balancing. |
| [processor.py](./processor.py) | `SupervisedProcessor`, which implements scaling, `fit`, and scoring. |
| [methods/](./methods/README.md) | Per-method parameter classes. |
