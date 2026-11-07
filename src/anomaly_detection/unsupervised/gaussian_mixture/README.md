# gaussian_mixture

## Overview

Negative log-likelihood under a Gaussian mixture fitted to the training samples, scored through `SampleScoringProcessor`, which scores the training samples out of fold. Unlike Mahalanobis distance, it models normal samples that form several clusters.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `GaussianMixtureParameters`, which validates the settings and builds the processor. |
| [covariance_structure.py](./covariance_structure.py) | `CovarianceStructure` enum of the covariance each component may have. |
