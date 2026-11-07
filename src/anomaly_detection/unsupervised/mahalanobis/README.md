# mahalanobis

## Overview

Mahalanobis distance from the training distribution.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `MahalanobisParameters`, which validates the settings and builds the processor. |
| [processor.py](./processor.py) | `MahalanobisProcessor`, using a shrinkage covariance estimate and out-of-fold training scores. |
