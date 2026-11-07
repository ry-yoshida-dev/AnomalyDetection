# prototype_similarity

## Overview

Compares samples with given embeddings of normal and anomalous, such as text embeddings from the same joint model as the samples. Prototypes of one kind that nearly cancel out, or normal and anomalous prototypes that point the same way, are rejected because they leave no direction to compare with.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `PrototypeSimilarityParameters`, which validates the prototypes and merges each kind into one unit vector. |
| [processor.py](./processor.py) | `PrototypeSimilarityProcessor`, which turns cosine similarities into an anomaly probability. |
