# methods

## Overview

Parameter classes for each zero-shot method. Every parameter class is frozen, validates its fields on construction, and builds a ready-to-use processor via `build_processor()`.

## Components

| Component | Description |
|-----------|-------------|
| [prototype_similarity.py](./prototype_similarity.py) | `PrototypeSimilarityParameters`. |
| [mutual_scoring.py](./mutual_scoring.py) | `MutualScoringParameters`. |
