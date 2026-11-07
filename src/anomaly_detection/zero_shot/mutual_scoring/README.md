# mutual_scoring

## Overview

Scores each sample against the other samples of the same batch, assuming anomalies are rare and unlike the rest. Scores and thresholds are relative to the batch, so they are not comparable across batches. With cosine distance, the default, zero vectors are rejected because they have no direction.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `MutualScoringParameters`, which validates the settings and builds the processor. |
| [processor.py](./processor.py) | `MutualScoringProcessor`, which scores samples by their distances to the nearest other samples. |
