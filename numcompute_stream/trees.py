"""
trees.py
--------
Streaming-compatible Decision Tree Classifier for NumCompute-Stream.

Supports incremental learning via .partial_fit(), Gini or entropy
impurity, and configurable depth/feature limits.

Author: NumCompute-Stream
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gini(y: np.ndarray) -> float:
    """Compute Gini impurity for label array y.

    Parameters
    ----------
    y : np.ndarray, shape (n,)

    Returns
    -------
    float : Gini impurity in [0, 0.5]
    """

#  <--------- impurity calculations --------->

    if y.size == 0:
        return 0.0
    
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    return float(1.0 - np.sum(probs ** 2))


# <--------- best split finding --------->

def _entropy(y: np.ndarray) -> float:
    """Compute Shannon entropy for label array y.

    Parameters
    ----------
    y : np.ndarray, shape (n,)

    Returns
    -------
    float : entropy >= 0
    """

    if y.size == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    probs = np.clip(probs, 1e-12, 1.0)

    return float(-np.sum(probs * np.log2(probs)))


# <--------- impurity calculations --------->

def _impurity(y: np.ndarray, criterion: str) -> float:
    """Dispatch to the correct impurity function."""

    if criterion == "gini":
        return _gini(y)
    elif criterion == "entropy":
        return _entropy(y)
    
    else:
        raise ValueError(f"Unknown criterion '{criterion}'. Use 'gini' or 'entropy'.")


# <--------- function for finding best split   --------->

def _best_split(X: np.ndarray, y: np.ndarray, criterion: str,
                max_features=None, rng=None):
    """Find the best (feature, threshold) split using vectorised operations.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    criterion : str
    max_features : int or None — number of features to consider
    rng : np.random.Generator or None

    Returns
    -------
    best_feat : int or None
    best_thresh : float or None
    best_gain : float
    """

    n_samples, n_features = X.shape

    if n_samples < 2:
        return None, None, 0.0


# <--------- here, we calculate the best split using vectorised operations --------->

    if rng is None:
        rng = np.random.default_rng()
    feat_indices = np.arange(n_features)

    if max_features is not None and max_features < n_features:
        feat_indices = rng.choice(n_features, size=max_features, replace=False)

    parent_impurity = _impurity(y, criterion)
    best_gain = 0.0
    best_feat = None
    best_thresh = None

    for feat in feat_indices:
        col = X[:, feat]

        if np.nanstd(col) == 0:
            continue

        unique_vals = np.unique(col[~np.isnan(col)])
        if unique_vals.size < 2:
            continue

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0


#  <--------- loop over features and thresholds to find the best split --------->

        for thresh in thresholds:
            mask = col <= thresh
            left_y = y[mask]
            right_y = y[~mask]

            if left_y.size == 0 or right_y.size == 0:
                continue

            n_l, n_r = left_y.size, right_y.size
            n_total = n_l + n_r
            gain = parent_impurity - (
                (n_l / n_total) * _impurity(left_y, criterion) +
                (n_r / n_total) * _impurity(right_y, criterion)
            )

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _Node:
    """Internal tree node."""

    __slots__ = ("feat", "thresh", "left", "right", "value", "n_samples")


#  <--------- simple class to represent a node in the decision tree --------->

    def __init__(self, feat=None, thresh=None, left=None,
                 right=None, value=None, n_samples=0):
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value       
        self.n_samples = n_samples

    @property
    def is_leaf(self):

        return self.value is not None


# ---------------------------------------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """Depth-limited decision tree with streaming support.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    criterion : str, default='gini'
        Impurity measure: 'gini' or 'entropy'.
    max_features : int or None, default=None
        Number of features to consider per split (used by Random Forest).
    random_state : int or None, default=None

    Examples
    --------

    >>> clf = DecisionTreeClassifier(max_depth=3)
    >>> clf.partial_fit(X_train, y_train)
    >>> preds = clf.predict(X_test)
    """


# <--------- main class for the decision tree classifier --------->

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 criterion: str = "gini", max_features=None,
                 random_state=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self._root = None
        self._classes = None
        self._n_features = None
        self._rng = np.random.default_rng(random_state)

        self._X_seen = None
        self._y_seen = None

    # ------------------------------------------------------------------
    # Private build helpers
    # ------------------------------------------------------------------


#  <--------- helper methods for building the tree and making predictions --------->

    def _leaf_value(self, y: np.ndarray):
        """Return majority class (tie-broken by smallest class label)."""
        classes, counts = np.unique(y, return_counts=True)

        return classes[np.argmax(counts)]


# <--------- helper method to compute the majority class for a leaf node --------->

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = X.shape[0]
        node = _Node(n_samples=n_samples)

        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                np.unique(y).size == 1):
            node.value = self._leaf_value(y)
            return node

        feat, thresh, gain = _best_split(
            X, y, self.criterion, self.max_features, self._rng
        )

        if feat is None or gain == 0.0:
            node.value = self._leaf_value(y)
            return node

        mask = X[:, feat] <= thresh
        node.feat = feat
        node.thresh = thresh
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)

        return node


# <--------- recursive method to build the tree based on the best splits --------->

    def _predict_one(self, x: np.ndarray, node: _Node):
        """Traverse tree for a single sample."""
        if node.is_leaf:
            return node.value
        if np.isnan(x[node.feat]) or x[node.feat] <= node.thresh:
            return self._predict_one(x, node.left)
        
        return self._predict_one(x, node.right)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


# <--------- public methods for fitting, predicting, and scoring the tree --------->

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Fit tree from scratch on (X, y).

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
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of rows. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        self._classes = np.unique(y)
        self._n_features = X.shape[1]
        self._X_seen = X.copy()
        self._y_seen = y.copy()
        self._root = self._build(X, y, depth=0)

        return self


# <--------- fit method to build the tree from scratch --------->

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    classes=None) -> "DecisionTreeClassifier":
        """Incrementally update tree with a new chunk of data.

        Accumulates data and rebuilds the tree. Supports streaming
        scenarios where data arrives chunk-by-chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        classes : array-like or None
            All possible classes (used on first call).

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.array(X, dtype=float))
        y = np.array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
            )

        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        if self._X_seen is None:
            self._X_seen = X
            self._y_seen = y
            if classes is not None:
                self._classes = np.array(classes)
            else:
                self._classes = np.unique(y)
            self._n_features = X.shape[1]

        else:
            if X.shape[1] != self._n_features:
                raise ValueError(
                    f"Expected {self._n_features} features, got {X.shape[1]}"
                )
            self._X_seen = np.vstack([self._X_seen, X])
            self._y_seen = np.concatenate([self._y_seen, y])
            self._classes = np.unique(self._y_seen)

        self._root = self._build(self._X_seen, self._y_seen, depth=0)

        return self


# <--------- partial_fit method to incrementally update the tree with new data --------->

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self._root is None:
            raise RuntimeError("Tree is not fitted yet. Call fit() or partial_fit() first.")
        X = np.atleast_2d(np.array(X, dtype=float))
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {X.shape[1]}"
            )
        
        return np.array([self._predict_one(x, self._root) for x in X])
    

# <--------- predict method to traverse the tree and return class labels --------->

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (soft predictions).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """

        if self._root is None:
            raise RuntimeError("Tree is not fitted yet.")
        X = np.atleast_2d(np.array(X, dtype=float))

        preds = self.predict(X)
        n_classes = len(self._classes)
        proba = np.zeros((X.shape[0], n_classes))

        class_to_idx = {c: i for i, c in enumerate(self._classes)}
        for i, p in enumerate(preds):
            if p in class_to_idx:
                proba[i, class_to_idx[p]] = 1.0

        return proba


# <--------- predict_proba method to return class probabilities --------->

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float : accuracy in [0, 1]
        """

        preds = self.predict(X)
        return float(np.mean(preds == np.array(y)))

    @property
    def classes_(self):
        """Unique class labels seen during training."""
        return self._classes

    @property
    def n_features_in_(self):
        """Number of input features."""
        return self._n_features


# <--------- score method to compute accuracy --------->

    def __repr__(self):
        return (
            f"DecisionTreeClassifier("
            f"max_depth={self.max_depth}, "
            f"criterion='{self.criterion}', "
            f"min_samples_split={self.min_samples_split})"
        )