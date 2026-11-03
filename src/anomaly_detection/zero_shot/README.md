# zero_shot

## Overview

Methods that need no training samples of the target, and therefore have no `fit`: each builds a ready-to-use `AnomalyProcessor`.

## Components

| Component | Description |
|-----------|-------------|
| [unit_normalizer.py](./unit_normalizer.py) | `UnitNormalizer`, which scales vectors for cosine similarity. |
| [methods/](./methods/README.md) | Per-method parameter classes. |
| [processors/](./processors/README.md) | Concrete processors, split by where the reference for normality comes from. |
