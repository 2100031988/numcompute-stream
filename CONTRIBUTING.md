# Contributing to NumCompute-Stream

Thank you for your interest in contributing to **NumCompute-Stream**!  
This document outlines the guidelines for contributing to this project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Guidelines](#development-guidelines)
- [Adding New Features](#adding-new-features)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Commit Guidelines](#commit-guidelines)

---

## Project Overview

NumCompute-Stream is a streaming, decision tree–based machine learning framework built on top of NumCompute. It supports incremental learning via `.partial_fit()`, ensemble methods (Bagging, Random Forest), and real-time metric visualisation.

**Allowed libraries:** NumPy and matplotlib only. No scikit-learn, pandas, or PyTorch.

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/2100031988/numcompute-stream.git
cd numcompute-stream

# 2. Install dependencies
pip install numpy matplotlib pytest

# 3. Run tests to confirm everything works
pytest tests/test_numcompute_stream.py -v
```

---

## Project Structure

```
numcompute-stream/
├── numcompute/              ← Base NumCompute package (do not modify)
├── numcompute_stream/       ← Main streaming framework
│   ├── __init__.py
│   ├── trees.py             ← DecisionTreeClassifier
│   ├── ensemble.py          ← BaggingClassifier, RandomForestClassifier
│   ├── streaming.py         ← StreamTrainer, chunk_data
│   ├── metrics.py           ← Streaming metrics
│   ├── pipeline.py          ← StreamingPipeline
│   └── visualise.py         ← Plotting functions
├── tests/
│   └── test_numcompute_stream.py
├── demo/
│   ├── stream_demo.ipynb
│   └── iris.csv
├── benchmark/
│   └── benchmark_streaming.py
└── README.md
```

---

## Development Guidelines

### Must Follow

- **NumPy only** — all core logic must use vectorised NumPy operations. Avoid Python loops unless absolutely necessary.
- **Streaming compatible** — every new component must expose `.partial_fit()` or `.update()` for chunk-wise updates.
- **Numerical stability** — handle NaNs, zero-variance features, empty chunks, and division-by-zero safely.
- **API consistency** — document input/output shapes clearly. Raise informative `ValueError` or `RuntimeError` for bad inputs.
- **No external ML libraries** — scikit-learn, pandas, PyTorch etc. are not allowed anywhere in the package.

### Do Not

- Modify the `numcompute/` base package
- Add dependencies outside NumPy and matplotlib
- Break existing tests when adding new features
- Use Python loops where NumPy vectorisation is possible

---

## Adding New Features

1. **Create your feature** in the appropriate module inside `numcompute_stream/`
2. **Export it** from `numcompute_stream/__init__.py`
3. **Write tests** — at least 3 unit tests per new class or function, including one edge case
4. **Update README.md** with usage example

Example — adding a new classifier:
```python
# numcompute_stream/trees.py
class MyNewClassifier:
    def __init__(self):
        self._fitted = False

    def partial_fit(self, X, y):
        # must support streaming
        ...
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Call partial_fit() first.")
        ...
```

---

## Running Tests

```bash
# Run all tests
pytest tests/test_numcompute_stream.py -v

# Run a specific test class
pytest tests/test_numcompute_stream.py::TestDecisionTreeClassifier -v

# Run a single test
pytest tests/test_numcompute_stream.py::TestNumericalStability::test_all_nan_column_handled -v

# Run with coverage
pytest tests/ --cov=numcompute_stream -v
```

All 65 existing tests must pass before submitting a contribution.

---

## Code Style

- Follow **PEP 8** conventions
- Use **docstrings** for all public classes and functions — include Parameters, Returns, and Examples sections
- Use meaningful variable names — no single letters except loop indices
- Add inline comments for non-obvious logic

Example docstring format:
```python
def partial_fit(self, X, y, classes=None):
    """Incrementally update model with a new chunk of data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    classes : array-like or None

    Returns
    -------
    self
    
    Examples
    --------
    >>> clf.partial_fit(X_chunk, y_chunk)
    """
```

---

## Commit Guidelines

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add RollingAccuracy metric with configurable window size"
git commit -m "Fix NaN imputation in partial_fit when entire column is NaN"
git commit -m "Add 5 unit tests for StreamingPipeline edge cases"

# Bad
git commit -m "fix stuff"
git commit -m "update"
```

---

## Questions

Open an issue on GitHub or contact the maintainer via the repository page.

Happy contributing! 
