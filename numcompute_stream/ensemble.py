"""
ensemble.py
-----------
Ensemble classifiers (Bagging & Random Forest) for NumCompute-Stream.

All ensembles support incremental .partial_fit() for streaming scenarios.

Author: NumCompute-Stream
"""

import numpy as np
from numcompute_stream.trees import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Base Ensemble
# ---------------------------------------------------------------------------

class _BaseEnsemble:
    """Shared interface for all ensemble methods."""

    def __init__(self, n_estimators: int, max_depth: int,
                 min_samples_split: int, criterion: str,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._estimators = []
        self._classes = None
        self._n_features = None

    def _majority_vote(self, predictions: np.ndarray) -> np.ndarray:
        """Return majority-vote predictions across estimators.

        Parameters
        ----------
        predictions : np.ndarray, shape (n_estimators, n_samples)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        n_samples = predictions.shape[1]
        result = np.empty(n_samples, dtype=predictions.dtype)
        for i in range(n_samples):
            col = predictions[:, i]
            classes, counts = np.unique(col, return_counts=True)
            result[i] = classes[np.argmax(counts)]
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by majority vote.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if not self._estimators:
            raise RuntimeError("Ensemble is not fitted. Call partial_fit() first.")
        X = np.atleast_2d(np.array(X, dtype=float))
        all_preds = np.array([est.predict(X) for est in self._estimators])
        return self._majority_vote(all_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict averaged class probabilities across all trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        if not self._estimators:
            raise RuntimeError("Ensemble is not fitted.")
        X = np.atleast_2d(np.array(X, dtype=float))
        n_classes = len(self._classes)
        proba_sum = np.zeros((X.shape[0], n_classes))
        for est in self._estimators:
            proba_sum += est.predict_proba(X)
        return proba_sum / len(self._estimators)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        return float(np.mean(self.predict(X) == np.array(y)))

    @property
    def classes_(self):
        return self._classes

    @property
    def estimators_(self):
        return self._estimators


# ---------------------------------------------------------------------------
# Bagging Classifier
# ---------------------------------------------------------------------------

class BaggingClassifier(_BaseEnsemble):
    """Bootstrap Aggregation (Bagging) ensemble of decision trees.

    Each tree is trained on a bootstrap sample of the accumulated data.
    Supports streaming via .partial_fit().

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the ensemble.
    max_depth : int, default=5
        Maximum depth per tree.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    criterion : str, default='gini'
        Impurity criterion: 'gini' or 'entropy'.
    max_samples : float, default=1.0
        Fraction of samples to draw per bootstrap.
    random_state : int or None, default=None

    Examples
    --------
    >>> clf = BaggingClassifier(n_estimators=10, max_depth=4)
    >>> clf.partial_fit(X_chunk, y_chunk)
    >>> preds = clf.predict(X_test)
    """

    def __init__(self, n_estimators: int = 10, max_depth: int = 5,
                 min_samples_split: int = 2, criterion: str = "gini",
                 max_samples: float = 1.0, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split,
                         criterion, random_state)
        if not 0.0 < max_samples <= 1.0:
            raise ValueError("max_samples must be in (0, 1]")
        self.max_samples = max_samples

        # Accumulate streaming data
        self._X_seen = None
        self._y_seen = None

    def _bootstrap(self, X: np.ndarray, y: np.ndarray):
        """Draw a bootstrap sample."""
        n = X.shape[0]
        n_draw = max(1, int(n * self.max_samples))
        idx = self._rng.integers(0, n, size=n_draw)
        return X[idx], y[idx]

    def _rebuild(self):
        """Rebuild all estimators from accumulated data."""
        X, y = self._X_seen, self._y_seen
        self._estimators = []
        seeds = self._rng.integers(0, 2**31, size=self.n_estimators)
        for seed in seeds:
            X_b, y_b = self._bootstrap(X, y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                random_state=int(seed)
            )
            tree.fit(X_b, y_b)
            self._estimators.append(tree)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """Fit ensemble from scratch.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.array(X, dtype=float))
        y = np.array(y)
        self._X_seen = X.copy()
        self._y_seen = y.copy()
        self._classes = np.unique(y)
        self._n_features = X.shape[1]
        self._rebuild()
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    classes=None) -> "BaggingClassifier":
        """Incrementally update ensemble with a new data chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        classes : array-like or None

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.array(X, dtype=float))
        y = np.array(y)

        # Handle NaNs
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        if self._X_seen is None:
            self._X_seen = X
            self._y_seen = y
            self._classes = np.array(classes) if classes is not None else np.unique(y)
            self._n_features = X.shape[1]
        else:
            if X.shape[1] != self._n_features:
                raise ValueError(
                    f"Feature count mismatch: expected {self._n_features}, got {X.shape[1]}"
                )
            self._X_seen = np.vstack([self._X_seen, X])
            self._y_seen = np.concatenate([self._y_seen, y])
            self._classes = np.unique(self._y_seen)

        self._rebuild()
        return self

    def __repr__(self):
        return (
            f"BaggingClassifier(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, criterion='{self.criterion}')"
        )


# ---------------------------------------------------------------------------
# Random Forest Classifier
# ---------------------------------------------------------------------------

class RandomForestClassifier(_BaseEnsemble):
    """Random Forest: Bagging + random feature subsampling per split.

    Each tree sees a bootstrap sample and considers only sqrt(n_features)
    features at each split, reducing correlation between trees.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees.
    max_depth : int, default=5
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split.
    criterion : str, default='gini'
        'gini' or 'entropy'.
    max_features : str or int or None, default='sqrt'
        Features to consider per split: 'sqrt', 'log2', int, or None (all).
    max_samples : float, default=1.0
        Bootstrap sample fraction.
    random_state : int or None, default=None

    Examples
    --------
    >>> rf = RandomForestClassifier(n_estimators=20, max_depth=5)
    >>> for chunk_X, chunk_y in stream:
    ...     rf.partial_fit(chunk_X, chunk_y)
    >>> preds = rf.predict(X_test)
    """

    def __init__(self, n_estimators: int = 10, max_depth: int = 5,
                 min_samples_split: int = 2, criterion: str = "gini",
                 max_features="sqrt", max_samples: float = 1.0,
                 random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split,
                         criterion, random_state)
        self.max_features = max_features
        self.max_samples = max_samples

        self._X_seen = None
        self._y_seen = None

    def _resolve_max_features(self, n_features: int) -> int:
        """Convert max_features spec to an integer."""
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError(
            f"Invalid max_features='{self.max_features}'. "
            "Use 'sqrt', 'log2', int, or None."
        )

    def _bootstrap(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        n_draw = max(1, int(n * self.max_samples))
        idx = self._rng.integers(0, n, size=n_draw)
        return X[idx], y[idx]

    def _rebuild(self):
        X, y = self._X_seen, self._y_seen
        n_features = X.shape[1]
        mf = self._resolve_max_features(n_features)
        self._estimators = []
        seeds = self._rng.integers(0, 2**31, size=self.n_estimators)
        for seed in seeds:
            X_b, y_b = self._bootstrap(X, y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=mf,
                random_state=int(seed)
            )
            tree.fit(X_b, y_b)
            self._estimators.append(tree)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit Random Forest from scratch.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.array(X, dtype=float))
        y = np.array(y)
        self._X_seen = X.copy()
        self._y_seen = y.copy()
        self._classes = np.unique(y)
        self._n_features = X.shape[1]
        self._rebuild()
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    classes=None) -> "RandomForestClassifier":
        """Incrementally update Random Forest with a new data chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        classes : array-like or None

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.array(X, dtype=float))
        y = np.array(y)

        # NaN imputation with column medians
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        if self._X_seen is None:
            self._X_seen = X
            self._y_seen = y
            self._classes = np.array(classes) if classes is not None else np.unique(y)
            self._n_features = X.shape[1]
        else:
            if X.shape[1] != self._n_features:
                raise ValueError(
                    f"Feature mismatch: expected {self._n_features}, got {X.shape[1]}"
                )
            self._X_seen = np.vstack([self._X_seen, X])
            self._y_seen = np.concatenate([self._y_seen, y])
            self._classes = np.unique(self._y_seen)

        self._rebuild()
        return self

    def feature_importances_(self) -> np.ndarray:
        """Compute mean impurity decrease across all trees per feature.

        Returns
        -------
        np.ndarray, shape (n_features,) — normalised importances
        """
        if not self._estimators:
            raise RuntimeError("Model not fitted.")
        # Approximate: count how often each feature is used as a split node
        counts = np.zeros(self._n_features)
        for tree in self._estimators:
            self._count_feature_uses(tree._root, counts)
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _count_feature_uses(self, node, counts):
        if node is None or node.is_leaf:
            return
        counts[node.feat] += 1
        self._count_feature_uses(node.left, counts)
        self._count_feature_uses(node.right, counts)

    def __repr__(self):
        return (
            f"RandomForestClassifier(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, max_features='{self.max_features}', "
            f"criterion='{self.criterion}')"
        )


# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------

EnsembleClassifier = RandomForestClassifier