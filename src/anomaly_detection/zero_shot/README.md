# zero_shot

## Overview

Methods that need no training samples of the target, and therefore have no `fit`: each builds a ready-to-use `AnomalyProcessor`.

## Components

| Component | Description |
|-----------|-------------|
| [unit_normalizer.py](./unit_normalizer.py) | `UnitNormalizer`, which scales vectors to unit length for cosine similarity. |
| [prototype_similarity/](./prototype_similarity/README.md) | Parameters and processor comparing samples with given embeddings of normal and anomalous. |
| [mutual_scoring/](./mutual_scoring/README.md) | Parameters and processor comparing samples with the rest of their batch. |
