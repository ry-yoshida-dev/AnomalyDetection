# sample_scoring

## Overview

A processor shared by estimators that score samples through `score_samples`, used by isolation forest and one-class SVM. Such estimators score the samples they were fitted on as more normal, so training samples are scored out of fold.

## Components

| Component | Description |
|-----------|-------------|
| [processor.py](./processor.py) | `SampleScoringProcessor`, which wraps any `SampleScorerLike` estimator. |
