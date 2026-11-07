# ensemble

## Overview

Combines several unsupervised methods by averaging the percentiles of their scores among their own out-of-fold training scores, so methods on different score scales weigh equally.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `EnsembleParameters`, which validates the members and builds the processor. |
| [processor.py](./processor.py) | `EnsembleProcessor`, which fits the members and averages their score percentiles. |
