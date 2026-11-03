# methods

## Overview

Parameter classes for each supervised method. Every parameter class is frozen, validates its fields on construction, and builds an unfitted processor via `build_processor()`.

## Components

| Component | Description |
|-----------|-------------|
| [logistic_regression.py](./logistic_regression.py) | `LogisticRegressionParameters`. |
| [random_forest.py](./random_forest.py) | `RandomForestParameters`. |
| [hist_gradient_boosting.py](./hist_gradient_boosting.py) | `HistGradientBoostingParameters`. |
