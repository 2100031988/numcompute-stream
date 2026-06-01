<h1 align="center"> NumCompute-Stream : A Machine Learning Framework </h1>


<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python"></a>
  <a href="https://numpy.org"><img src="https://img.shields.io/badge/NumPy-Enabled-orange?logo=numpy" alt="NumPy"></a>
  <a href="https://matplotlib.org"><img src="https://img.shields.io/badge/Matplotlib-Enabled-blue?logo=matplotlib" alt="Matplotlib"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-lightgrey" alt="License"></a>
</p>


<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#api-reference">API</a> •
  <a href="#benchmarking">Benchmarking</a> •
  <a href="#running-tests">Tests</a>
</p>

---

## Overview

**NumCompute** is the core package which is a lightweight machine learning framework built using **NumPy** and can function on libraries such as **scikit-learn**.

**NumCompute-Stream** It extends numcompute with an additionally python library which includes **numpy** and also **matplotlib**.

There are numerous usages of this library but some listed below:
  1. It is a perfect tool for real-time and incremental learning where data arrives in chunks rather than all at once.
  2. We can learn how ensemble methods like Bagging and Random Forest work from scratch using only NumPy.
  3. Lastly, the most important usage is streaming pipelines where we can chain transformers and estimators and watch model performance evolve chunk by chunk.

---

## Installation

### Install from Source

We can install the projec from my repository through github commands in the **terminal** in visual studio code.

```bash
git clone https://github.com/2100031988/numcompute-stream.git
cd numcompute-stream
pip install numpy matplotlib
```
---

## Project Structure

```
numcompute-stream/
├── numcompute/                         
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
├── numcompute_stream/                  
│   ├── __init__.py
│   ├── trees.py                        
│   ├── ensemble.py                     
│   ├── streaming.py                   
│   ├── metrics.py                     
│   ├── pipeline.py                     
│   └── visualise.py                    
│
├── tests/
│   ├── test_numcompute.py
│   └── test_numcompute_stream.py       
│
├── demo/
│   ├── quickstart.ipynb
│   └── stream_demo.ipynb               
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

## API Reference

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

</br>

### § 2 &nbsp;·&nbsp; `numcompute_stream` — Streaming Framework

</br>

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

Plots generated automatically: `tree_accuracy.png`, `model_comparison.png`, `predictions_vs_truth.png`, `streaming_metrics.png`, 
`confusion_matrix.png`, `feature_importances.png`, `benchmark.png`

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
