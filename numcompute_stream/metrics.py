"""
metrics.py
----------
Streaming-compatible classification metrics for NumCompute-Stream.

All metrics support:
    - .update(y_true_chunk, y_pred_chunk) for incremental updates
    - .reset() to clear state
    - .result() to get current metric value

Author: NumCompute-Stream
"""

import numpy as np


# ---------------------------------------------------------------------------
# Base streaming metric
# ---------------------------------------------------------------------------

class _StreamingMetric:
    """Abstract base for all streaming metrics."""

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def result(self) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

class StreamingAccuracy(_StreamingMetric):
    """Cumulative accuracy over streaming chunks.

    Examples
    --------
    >>> acc = StreamingAccuracy()
    >>> acc.update(y_true_chunk, y_pred_chunk)
    >>> print(acc.result())
    """


# <------- Cumulative accuracy implementation ------->

    def __init__(self):
        self._correct = 0
        self._total = 0
        self._history = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Accumulate correct predictions.

        Parameters
        ----------
        y_true : array-like
        y_pred : array-like
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        
        self._correct += int(np.sum(y_true == y_pred))
        self._total += len(y_true)
        self._history.append(self.result())


# <------- End of cumulative accuracy implementation ------->

    def reset(self):
        """Clear all accumulated state."""
        self._correct = 0
        self._total = 0
        self._history = []

    def result(self) -> float:
        """Return cumulative accuracy.

        Returns
        -------
        float : accuracy in [0, 1], or 0.0 if no data seen.
        """
        return self._correct / self._total if self._total > 0 else 0.0

    @property                                   # Optional: expose history for analysis
    def history(self) -> np.ndarray:
        """Per-chunk cumulative accuracy history."""
        return np.array(self._history)

    def __repr__(self):
        return f"StreamingAccuracy(result={self.result():.4f}, n={self._total})"


# ---------------------------------------------------------------------------
# Precision, Recall, F1 (binary and macro)
# ---------------------------------------------------------------------------

class StreamingPrecisionRecallF1(_StreamingMetric):
    """Cumulative Precision, Recall, and F1 over streaming chunks.

    Supports binary (pos_label) and macro-averaged multi-class.

    Parameters
    ----------
    average : str, default='macro'
        'binary' or 'macro'.
    pos_label : int or str, default=1
        Positive class for binary mode.

    Examples
    --------
    >>> prf = StreamingPrecisionRecallF1(average='macro')
    >>> prf.update(y_true, y_pred)
    >>> print(prf.result())
    """

# <------- Cumulative precision/recall/F1 implementation ------->

    def __init__(self, average: str = "macro", pos_label=1):
        if average not in ("binary", "macro"):
            raise ValueError("average must be 'binary' or 'macro'")
        
        self.average = average
        self.pos_label = pos_label
        self._y_true_all = []
        self._y_pred_all = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        self._y_true_all.extend(np.array(y_true).tolist())
        self._y_pred_all.extend(np.array(y_pred).tolist())

    def reset(self):
        self._y_true_all = []
        self._y_pred_all = []


# <------- Result implementation ------->

    def result(self) -> dict:
        """Return precision, recall, f1.

        Returns
        -------
        dict with keys 'precision', 'recall', 'f1'
        """
        yt = np.array(self._y_true_all)
        yp = np.array(self._y_pred_all)
        if yt.size == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if self.average == "binary":
            return self._binary(yt, yp, self.pos_label)
        return self._macro(yt, yp)

    @staticmethod

    def _binary(yt, yp, pos_label) -> dict:
        tp = np.sum((yp == pos_label) & (yt == pos_label))
        fp = np.sum((yp == pos_label) & (yt != pos_label))
        fn = np.sum((yp != pos_label) & (yt == pos_label))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


# <------- Macro precision/recall/F1 implementation ------->

    @staticmethod
    def _macro(yt, yp) -> dict:
        classes = np.unique(yt)
        precs, recs, f1s = [], [], []

        for cls in classes:
            tp = np.sum((yp == cls) & (yt == cls))
            fp = np.sum((yp == cls) & (yt != cls))
            fn = np.sum((yp != cls) & (yt == cls))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

        return {
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "f1": float(np.mean(f1s)),
        }

    def __repr__(self):
        r = self.result()

        return (
            f"StreamingPrecisionRecallF1("
            f"precision={r['precision']:.4f}, "
            f"recall={r['recall']:.4f}, "
            f"f1={r['f1']:.4f})"
        )


# ---------------------------------------------------------------------------
# Confusion Matrix (cumulative)
# ---------------------------------------------------------------------------

class StreamingConfusionMatrix(_StreamingMetric):
    """Cumulative confusion matrix over streaming data.

    Parameters
    ----------
    classes : array-like or None — if None, inferred from data.

    Examples
    --------
    >>> cm = StreamingConfusionMatrix(classes=[0, 1, 2])
    >>> cm.update(y_true, y_pred)
    >>> print(cm.result())
    """


# <------- Cumulative confusion matrix implementation ------->

    def __init__(self, classes=None):
        self.classes = np.array(classes) if classes is not None else None
        self._matrix = None


# <------- Update implementation ------->

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if self.classes is None:
            self.classes = np.unique(np.concatenate([y_true, y_pred]))

        n = len(self.classes)
        if self._matrix is None:
            self._matrix = np.zeros((n, n), dtype=int)

        c2i = {c: i for i, c in enumerate(self.classes)}
        for t, p in zip(y_true, y_pred):
            if t in c2i and p in c2i:
                self._matrix[c2i[t], c2i[p]] += 1

    def reset(self):
        self._matrix = None


# <------- Result implementation ------->

    def result(self) -> np.ndarray:
        """Return accumulated confusion matrix.

        Returns
        -------
        np.ndarray, shape (n_classes, n_classes)
        """
        if self._matrix is None:
            return np.array([[]])
        
        return self._matrix.copy()

    def __repr__(self):
        shape = self._matrix.shape if self._matrix is not None else (0, 0)

        return f"StreamingConfusionMatrix(shape={shape})"


# ---------------------------------------------------------------------------
# Rolling window accuracy
# ---------------------------------------------------------------------------

class RollingAccuracy(_StreamingMetric):
    """Accuracy computed over a sliding window of recent samples.

    Parameters
    ----------
    window_size : int, default=200
        Number of recent samples to keep in the window.

    Examples
    --------
    >>> ra = RollingAccuracy(window_size=100)
    >>> ra.update(y_true_chunk, y_pred_chunk)
    >>> print(ra.result())
    """


# <------- Rolling window accuracy implementation ------->

    def __init__(self, window_size: int = 200):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        
        self.window_size = window_size
        self._buffer_true = []
        self._buffer_pred = []


# <------- Update implementation ------->

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):

        y_true = list(np.array(y_true))
        y_pred = list(np.array(y_pred))
        self._buffer_true.extend(y_true)
        self._buffer_pred.extend(y_pred)

        if len(self._buffer_true) > self.window_size:
            self._buffer_true = self._buffer_true[-self.window_size:]
            self._buffer_pred = self._buffer_pred[-self.window_size:]

    def reset(self):
        self._buffer_true = []
        self._buffer_pred = []


# <------- Result implementation ------->

    def result(self) -> float:
        """Return accuracy over the current window.

        Returns
        -------
        float
        """
        if not self._buffer_true:
            return 0.0
        
        yt = np.array(self._buffer_true)
        yp = np.array(self._buffer_pred)

        return float(np.mean(yt == yp))

    def __repr__(self):
        return (
            f"RollingAccuracy(window={self.window_size}, "
            f"result={self.result():.4f}, "
            f"n_in_window={len(self._buffer_true)})"
        )


# ---------------------------------------------------------------------------
# Convenience functions (stateless)
# ---------------------------------------------------------------------------


# <------- Accuracy convenience function implementation ------->

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy for a single batch.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    yt = np.array(y_true)
    yp = np.array(y_pred)

    if yt.size == 0:
        return 0.0
    
    return float(np.mean(yt == yp))


# <------- Precision/Recall/F1 convenience function implementation ------->

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                         average: str = "macro", pos_label=1) -> dict:
    
    """Compute precision, recall, F1 for a single batch.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    average : str, 'macro' or 'binary'
    pos_label : int or str — for binary mode

    Returns
    -------
    dict with keys 'precision', 'recall', 'f1'
    """

    metric = StreamingPrecisionRecallF1(average=average, pos_label=pos_label)
    metric.update(y_true, y_pred)

    return metric.result()


# <------- Confusion matrix convenience function implementation ------->

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     classes=None) -> np.ndarray:
    """Compute confusion matrix for a single batch.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    classes : array-like or None

    Returns
    -------
    np.ndarray, shape (n_classes, n_classes)
    """

    cm = StreamingConfusionMatrix(classes=classes)
    cm.update(y_true, y_pred)
    
    return cm.result()