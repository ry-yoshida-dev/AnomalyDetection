# k_nearest_neighbors

## Overview

Mean distance to the k nearest training samples.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `KNearestNeighborsParameters`, which validates the settings and builds the processor. |
| [processor.py](./processor.py) | `KNearestNeighborsProcessor`, using training distances that exclude each sample. |
