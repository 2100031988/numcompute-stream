# NumCompute & NumCompute-Stream


<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python"></a>
  <a href="https://numpy.org"><img src="https://img.shields.io/badge/NumPy-Enabled-orange?logo=numpy" alt="NumPy"></a>
  <a href="https://matplotlib.org"><img src="https://img.shields.io/badge/Matplotlib-Enabled-blue?logo=matplotlib" alt="Matplotlib"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License"></a>
</p>


<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#api-reference">API</a> •
  <a href="#benchmarking">Benchmarking</a> •
  <a href="#running-tests">Tests</a>
</p>

---

## Demo

<p align="center">
  <img src="demo/demo.gif" alt="NumCompute Demo" width="600">
</p>

<p align="center">
  <em><a href="https://www.youtube.com/watch?v=zixmd_Ht3O0&t=1s">Watch the full explanation here!</a></em>
</p>

---

## Overview

**NumCompute** is the core package — a transparent, NumPy-only ML toolkit perfect for learning algorithms from scratch, evaluating pipelines, and building experiments.

**NumCompute-Stream** extends it with:

- **Streaming Learning** — all components support `.partial_fit()` for chunk-by-chunk updates
- **Ensemble Methods** — Bagging and Random Forest built from decision trees
- **Built-in Visualisation** — matplotlib-based plotting for metrics, comparisons, and predictions
- **Streaming Metrics** — cumulative accuracy, precision/recall/F1, confusion matrix, rolling accuracy
- **Streaming Pipeline** — chain transformers and estimators with a shared `.partial_fit()` interface

---

## Project Structure

```
numcompute-stream/
├── numcompute/                         ← Core NumCompute package
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   └── utils.py
│
├── numcompute_stream/                  ← Streaming framework
│   ├── __init__.py
│   ├── trees.py                        ← DecisionTreeClassifier (Gini/Entropy, partial_fit)
│   ├── ensemble.py                     ← BaggingClassifier, RandomForestClassifier
│   ├── streaming.py                    ← StreamTrainer, chunk_data
│   ├── metrics.py                      ← Streaming metrics (accuracy, PRF1, CM, rolling)
│   ├── pipeline.py                     ← StreamingPipeline
│   └── visualise.py                    ← Plotting functions
│
├── tests/
│   ├── test_numcompute.py
│   └── test_numcompute_stream.py       ← 30+ unit tests
│
├── demo/
│   ├── quickstart.ipynb
│   └── stream_demo.ipynb               ← Full streaming demo notebook
│
├── benchmark/
│   ├── benchmarking.py
│   └── benchmark_streaming.py
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Installation

### Option 1: Install from TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ numcompute
```

### Option 2: Install from Source

```bash
git clone https://github.com/2100031988/numcompute-stream.git
cd numcompute-stream
pip install numpy matplotlib pytest
```

---

## Quick Start

### 1 · NumCompute (Core)

```python
import numcompute as nc

data = [10, 20, 30, 40, 50]

print("Mean:",       nc.mean(data))
print("Median:",     nc.median(data))
print("Std:",        nc.std(data))
print("Normalized:", nc.normalize(data))
print("Sorted:",     nc.sort(data))
print("Search 30:",  nc.binary_search(data, 30))
print("MSE:",        nc.mse([1, 2, 3], [1, 2, 4]))
```

### 2 · NumCompute-Stream (Streaming)

```python
import numpy as np
from numcompute_stream.trees import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.streaming import StreamTrainer, chunk_data
from numcompute_stream import visualise

# Iris dataset — 150 samples, 4 features, 10 chunks of 15
X, y = load_iris_data()

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
rf   = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)

t1 = StreamTrainer(tree, verbose=True)
t2 = StreamTrainer(rf,   verbose=True)

for X_chunk, y_chunk in chunk_data(X, y, chunk_size=15):
    t1.fit_chunk(X_chunk, y_chunk)
    t2.fit_chunk(X_chunk, y_chunk)

# Decision Tree  → 0.8600 accuracy
# Random Forest  → 0.8133 accuracy

visualise.compare_models(
    t1.get_metric_history("metric"),
    t2.get_metric_history("metric"),
    labels=("Decision Tree", "Random Forest")
)
```

### 3 · Pipeline

```python
from numcompute_stream.pipeline import StreamingPipeline
from numcompute.preprocessing import StandardScaler

pipe = StreamingPipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(n_estimators=5, max_depth=3)),
])

for X_chunk, y_chunk in chunk_data(X, y, chunk_size=15):
    pipe.partial_fit(X_chunk, y_chunk)

# Pipeline → 0.8400 accuracy  (plots saved automatically)
```

---

## API Reference

<br>

### § 1 &nbsp;·&nbsp; `numcompute` — Core Package

| Module | Function | Description |
|---|---|---|
| `stats` | `mean(data)` | Arithmetic mean |
| `stats` | `median(data)` | Median value |
| `stats` | `std(data)` | Standard deviation |
| `preprocessing` | `normalize(data)` | Min-max normalisation |
| `preprocessing` | `StandardScaler()` | Zero-mean, unit-variance scaling |
| `preprocessing` | `LabelEncoder()` | Encodes categorical labels |
| `sort_search` | `sort(data)` | Sorts an array |
| `sort_search` | `binary_search(data, target)` | Binary search on sorted array |
| `metrics` | `mse(y_true, y_pred)` | Mean Squared Error |
| `pipeline` | `Pipeline(steps)` | Chains transformers sequentially |
| `pipeline` | `.fit(X)` | Fits the pipeline to data |
| `pipeline` | `.transform(X)` | Applies transformations to data |

<br>

---

### § 2 &nbsp;·&nbsp; `numcompute_stream` — Streaming Framework

<br>

#### § 2.1 &nbsp; `numcompute_stream.trees`

| Class | Description |
|---|---|
| `DecisionTreeClassifier` | Depth-limited tree with Gini/entropy, `partial_fit()` |

<br>

#### § 2.2 &nbsp; `numcompute_stream.ensemble`

| Class | Description |
|---|---|
| `BaggingClassifier` | Bootstrap aggregation of decision trees |
| `RandomForestClassifier` | Bagging + random feature subsampling |
| `EnsembleClassifier` | Alias for `RandomForestClassifier` |

<br>

#### § 2.3 &nbsp; `numcompute_stream.streaming`

| Name | Description |
|---|---|
| `StreamTrainer` | Manages chunk-wise training, logging, scoring |
| `chunk_data(X, y, chunk_size)` | Splits arrays into chunks for streaming |

<br>

#### § 2.4 &nbsp; `numcompute_stream.metrics`

| Class / Function | Description |
|---|---|
| `StreamingAccuracy` | Cumulative accuracy with `.update()` / `.result()` |
| `StreamingPrecisionRecallF1` | Macro/binary PRF1 over streaming data |
| `StreamingConfusionMatrix` | Accumulating confusion matrix |
| `RollingAccuracy` | Sliding-window accuracy |
| `accuracy(y_true, y_pred)` | Stateless batch accuracy |

<br>

#### § 2.5 &nbsp; `numcompute_stream.visualise`

| Function | Description |
|---|---|
| `plot_metric_over_time(values, title, ylabel)` | Metric vs chunk index |
| `compare_models(m1, m2, labels)` | Two-model metric comparison |
| `plot_predictions_vs_ground_truth(y_true, y_pred)` | Correct/wrong scatter |
| `plot_confusion_matrix(y_true, y_pred)` | Heatmap confusion matrix |
| `plot_memory_over_time(logs)` | Memory footprint per chunk |
| `plot_feature_importances(importances)` | Feature importance bar chart |

<br>

---

## Benchmarking

```bash
# Core benchmarks
python benchmark/benchmarking.py

# Streaming benchmarks
python benchmark/benchmark_streaming.py
```

### Core — Vectorised vs Loop

The core benchmarks show vectorised NumPy is approximately **80× faster** than loop-based operations with identical accuracy:

| Method | Mean | Time (seconds) |
|---|---|---|
| Vectorised | 0.5001100158230776 | 0.001165 |
| Loop-based | 0.5001100158230763 | 0.093470 |

### Streaming — Model Comparison on Iris

Results across 10 streaming chunks (150 samples, 4 features):

| Model | Accuracy | Avg per chunk | Total time |
|---|---|---|---|
| Decision Tree (depth=3) | 0.8600 | 9.79 ms | 0.098 s |
| Decision Tree (depth=5) | 0.8600 | 9.20 ms | 0.092 s |
| Bagging (n=5, depth=3) | 0.8267 | 34.16 ms | 0.342 s |
| Random Forest (n=5, depth=3) | 0.8000 | 18.85 ms | 0.188 s |
| Random Forest (n=10, depth=5) | 0.8133 | 51.97 ms | 0.520 s |

Plots generated automatically: `tree_accuracy.png`, `model_comparison.png`, `predictions_vs_truth.png`, `streaming_metrics.png`, `confusion_matrix.png`, `feature_importances.png`, `benchmark.png`

---

## Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=numcompute_stream -v
```

---

## Contributors

**Sabyasachi Kumar**

---

## License

This project is licensed under the [GPL License](LICENSE).
