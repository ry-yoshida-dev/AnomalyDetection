# prototype_similarity

## Overview

Compares samples with given embeddings of normal and anomalous, such as text embeddings from the same joint model as the samples.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `PrototypeSimilarityParameters`, which validates the prototypes and merges each kind into one unit vector. |
| [processor.py](./processor.py) | `PrototypeSimilarityProcessor`, which turns cosine similarities into an anomaly probability. |
