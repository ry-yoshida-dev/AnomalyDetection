# processors

## Overview

Concrete `AnomalyProcessor` implementations that decide without fitting.

## Components

| Component | Description |
|-----------|-------------|
| [prototype_similarity.py](./prototype_similarity.py) | `PrototypeSimilarityProcessor`, which compares samples with given embeddings of normal and anomalous. |
| [mutual_scoring.py](./mutual_scoring.py) | `MutualScoringProcessor`, which compares samples with the rest of their batch. |
