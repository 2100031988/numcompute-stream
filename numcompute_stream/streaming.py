"""
streaming.py
------------
StreamTrainer: manages models, pipeline, and per-chunk logging
for online/streaming machine learning with NumCompute-Stream.

Author: NumCompute-Stream
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# StreamTrainer
# ---------------------------------------------------------------------------

class StreamTrainer:
    """Manages incremental training of a model over streaming data chunks.

    Wraps any model with .partial_fit() and .predict() methods, logging
    per-chunk accuracy, loss, and memory footprint.

    Parameters
    ----------
    model : object
        Any model with .partial_fit(X, y) and .predict(X) methods.
        e.g. DecisionTreeClassifier, RandomForestClassifier.
    preprocessor : object or None, default=None
        Transformer with .partial_fit(X) and .transform(X). Applied
        before passing data to model.
    metric_fn : callable or None, default=None
        Custom metric function: metric_fn(y_true, y_pred) -> float.
        Defaults to accuracy.
    verbose : bool, default=True
        Whether to print per-chunk logs.

    Attributes
    ----------
    logs_ : list of dict
        Per-chunk records with keys: chunk, n_samples, accuracy,
        cumulative_accuracy, time_s, memory_bytes.

    Examples
    --------
    >>> trainer = StreamTrainer(model=RandomForestClassifier())
    >>> for X_chunk, y_chunk in stream:
    ...     trainer.fit_chunk(X_chunk, y_chunk)
    >>> print(trainer.logs_)
    """

    def __init__(self, model, preprocessor=None, metric_fn=None,
                 verbose: bool = True):
        self.model = model
        self.preprocessor = preprocessor
        self.metric_fn = metric_fn if metric_fn is not None else self._accuracy
        self.verbose = verbose

        self.logs_ = []
        self._chunk_idx = 0
        self._total_correct = 0
        self._total_seen = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default metric: accuracy."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.size == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    def _memory_of(self, obj) -> int:
        """Rough memory estimate in bytes using numpy array sizes."""
        total = 0
        # Check for accumulated data arrays in common attribute names
        for attr in ("_X_seen", "_y_seen"):
            arr = getattr(obj, attr, None)
            if isinstance(arr, np.ndarray):
                total += arr.nbytes
        return total

    def _preprocess(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Apply preprocessor if present."""
        if self.preprocessor is None:
            return X
        if fit:
            self.preprocessor.partial_fit(X)
        return self.preprocessor.transform(X)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_chunk(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Incrementally train on one chunk and log metrics.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        dict : log record for this chunk
        """
        X = np.atleast_2d(np.array(X, dtype=float))
        y = np.array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
            )

        t0 = time.perf_counter()

        # Preprocess
        X_proc = self._preprocess(X, fit=True)

        # Predict BEFORE updating (test on unseen chunk)
        chunk_metric = 0.0
        if self._chunk_idx > 0:
            try:
                y_pred = self.model.predict(X_proc)
                chunk_metric = self.metric_fn(y, y_pred)
                self._total_correct += int(np.sum(y == y_pred))
            except Exception:
                pass

        # Update model
        self.model.partial_fit(X_proc, y)
        elapsed = time.perf_counter() - t0

        self._total_seen += X.shape[0]
        cumulative_acc = (
            self._total_correct / self._total_seen
            if self._total_seen > 0 else 0.0
        )

        mem = self._memory_of(self.model)

        record = {
            "chunk": self._chunk_idx,
            "n_samples": X.shape[0],
            "metric": chunk_metric,
            "cumulative_metric": cumulative_acc,
            "time_s": round(elapsed, 5),
            "memory_bytes": mem,
        }
        self.logs_.append(record)

        if self.verbose:
            print(
                f"[Chunk {self._chunk_idx:3d}] "
                f"n={X.shape[0]:5d} | "
                f"metric={chunk_metric:.4f} | "
                f"cumulative={cumulative_acc:.4f} | "
                f"time={elapsed:.4f}s | "
                f"mem={mem/1024:.1f}KB"
            )

        self._chunk_idx += 1
        return record

    def score_chunk(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate current model on a held-out chunk without updating.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float : metric score
        """
        X = np.atleast_2d(np.array(X, dtype=float))
        y = np.array(y)
        X_proc = self._preprocess(X, fit=False)
        y_pred = self.model.predict(X_proc)
        return self.metric_fn(y, y_pred)

    def get_metric_history(self, key: str = "metric") -> np.ndarray:
        """Extract a metric history from logs.

        Parameters
        ----------
        key : str, default='metric'
            Key from log records: 'metric', 'cumulative_metric',
            'time_s', 'memory_bytes'.

        Returns
        -------
        np.ndarray, shape (n_chunks,)
        """
        if not self.logs_:
            return np.array([])
        return np.array([rec.get(key, 0.0) for rec in self.logs_])

    def reset(self):
        """Reset logs and chunk counter (model state is preserved)."""
        self.logs_ = []
        self._chunk_idx = 0
        self._total_correct = 0
        self._total_seen = 0

    def summary(self) -> dict:
        """Return a summary of all chunks.

        Returns
        -------
        dict with keys: n_chunks, total_samples, mean_metric,
        final_cumulative_metric, total_time_s
        """
        if not self.logs_:
            return {}
        metrics = self.get_metric_history("metric")
        times = self.get_metric_history("time_s")
        return {
            "n_chunks": len(self.logs_),
            "total_samples": self._total_seen,
            "mean_metric": float(np.mean(metrics[1:])) if len(metrics) > 1 else 0.0,
            "final_cumulative_metric": self.logs_[-1].get("cumulative_metric", 0.0),
            "total_time_s": float(np.sum(times)),
        }

    def __repr__(self):
        return (
            f"StreamTrainer(model={self.model.__class__.__name__}, "
            f"chunks_seen={self._chunk_idx})"
        )


# ---------------------------------------------------------------------------
# Data chunking utility
# ---------------------------------------------------------------------------

def chunk_data(X: np.ndarray, y: np.ndarray,
               chunk_size: int, shuffle: bool = False,
               random_state=None):
    """Split arrays into chunks for streaming simulation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    chunk_size : int — number of samples per chunk
    shuffle : bool, default=False — shuffle before chunking
    random_state : int or None

    Yields
    ------
    (X_chunk, y_chunk) tuples

    Examples
    --------
    >>> for X_c, y_c in chunk_data(X, y, chunk_size=100):
    ...     trainer.fit_chunk(X_c, y_c)
    """
    X = np.atleast_2d(np.array(X, dtype=float))
    y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    for start in range(0, n, chunk_size):
        idx = indices[start: start + chunk_size]
        yield X[idx], y[idx]