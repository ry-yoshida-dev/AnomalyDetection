# empirical_cumulative_distribution

## Overview

ECOD (empirical cumulative distribution-based outlier detection): how far into the tails of each feature's training distribution a sample lies, summed over features. It has no hyperparameters and is unaffected by monotonic feature scaling.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `EmpiricalCumulativeDistributionParameters`, which validates the settings and builds the processor. |
| [processor.py](./processor.py) | `EmpiricalCumulativeDistributionProcessor`, using tail probabilities that leave each training sample out. |
