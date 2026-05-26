import numpy as np

from numcompute.io import load_csv

from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer
)

from numcompute.stats import mean, median, std, quantile

from numcompute.sort_search import topk, binary_search, quickselect

from numcompute.rank import rank, percentile

from numcompute.metrics import (
    accuracy, precision, recall, f1_score, confusion_matrix, mse
)

from numcompute.optim import grad


# =========================
# IO TESTS
# =========================
def test_load_csv_basic():
    data = load_csv("tests/sample.csv", delimiter=",")
    assert isinstance(data, np.ndarray)


# =========================
# PREPROCESSING TESTS
# =========================
def test_standard_scaler():
    X = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    assert np.allclose(np.mean(Xs, axis=0), 0, atol=1e-7)


def test_minmax_scaler():
    X = np.array([[1], [2], [3]])
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    assert np.isclose(Xs.min(), 0)
    assert np.isclose(Xs.max(), 1)


def test_onehot_encoder():
    X = np.array([["A"], ["B"], ["A"]])
    enc = OneHotEncoder()
    Xo = enc.fit_transform(X)
    assert Xo.shape[1] == 2


def test_imputer_mean():
    X = np.array([[1, np.nan], [3, 4]])
    imp = SimpleImputer()
    Xf = imp.fit_transform(X)
    assert not np.isnan(Xf).any()


# =========================
# STATS TESTS
# =========================
def test_mean():
    assert np.isclose(mean([1, 2, 3]), 2)


def test_median():
    assert np.isclose(median([1, 2, 3]), 2)


def test_std():
    assert std([1, 1, 1]) == 0


def test_quantile():
    assert 1 <= quantile([1, 2, 3], 50) <= 3


# =========================
# SORT & SEARCH TESTS
# =========================

def test_topk():
    v = np.array([1, 5, 3, 2])
    top, idx = topk(v, 2)
    assert len(top) == 2


def test_binary_search_found():
    arr = np.array([1, 2, 3, 4])
    idx, found = binary_search(arr, 3)
    assert found


def test_binary_search_not_found():
    arr = np.array([1, 2, 3])
    idx, found = binary_search(arr, 5)
    assert not found


def test_quickselect():
    arr = np.array([3, 1, 2])
    assert quickselect(arr, 1) == 2


# =========================
# RANK TESTS
# =========================

def test_rank():
    r = rank([10, 20, 20], method="dense")
    assert len(r) == 3


def test_percentile():
    assert 0 <= percentile([1, 2, 3], 50) <= 3


# =========================
# METRICS TESTS
# =========================
def test_accuracy():
    assert accuracy([1, 0], [1, 0]) == 1


def test_precision_recall():
    y = [1, 0, 1]
    yhat = [1, 0, 0]
    assert precision(y, yhat) >= 0
    assert recall(y, yhat) >= 0


def test_f1():
    y = [1, 0, 1]
    yhat = [1, 0, 0]
    assert 0 <= f1_score(y, yhat) <= 1


def test_confusion_matrix():
    cm = confusion_matrix([0, 1], [0, 1])
    assert cm.shape == (2, 2)


def test_mse():
    assert mse([1, 2], [1, 2]) == 0


# =========================
# OPTIM TESTS
# =========================

def test_grad():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    g = grad(f, np.array([1.0, 2.0]))
    assert g.shape == (2,)