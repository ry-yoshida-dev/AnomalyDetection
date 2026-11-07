# principal_component

## Overview

Reconstruction error from the principal subspace of the training samples: how far a sample lies outside the directions along which normal samples vary.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `PrincipalComponentParameters`, which validates the settings and builds the processor. |
| [processor.py](./processor.py) | `PrincipalComponentProcessor`, using out-of-fold training scores. |
| [principal_subspace.py](./principal_subspace.py) | `PrincipalSubspace`, the fitted mean and leading components, and the residuals from them. |
