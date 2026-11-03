# Tests

## Overview

Unit tests for the `anomaly_detection` package, one file per family of methods.

## Components

| Component | Description |
| --- | --- |
| [`data.py`](data.py) | `SyntheticFeatures`, the shared normal and anomalous samples. |
| [`test_package.py`](test_package.py) | Tests for the public API layout and feature validation. |
| [`test_supervised.py`](test_supervised.py) | Tests for `anomaly_detection.supervised`. |
| [`test_unsupervised.py`](test_unsupervised.py) | Tests for `anomaly_detection.unsupervised`. |
| [`test_zero_shot.py`](test_zero_shot.py) | Tests for `anomaly_detection.zero_shot`. |

## Examples

```bash
uv run pytest
```
