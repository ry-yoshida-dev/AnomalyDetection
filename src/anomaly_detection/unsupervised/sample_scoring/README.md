# sample_scoring

## Overview

A processor shared by estimators whose `score_samples` is not biased on their training samples, used by isolation forest and one-class SVM.

## Components

| Component | Description |
|-----------|-------------|
| [processor.py](./processor.py) | `SampleScoringProcessor`, which wraps any `SampleScorerLike` estimator. |
