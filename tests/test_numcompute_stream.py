"""
test_numcompute_stream.py
--------------------------
Unit tests for NumCompute-Stream framework.

Covers:
  - DecisionTreeClassifier (standard + edge cases + streaming)
  - BaggingClassifier
  - RandomForestClassifier
  - StreamTrainer + chunk_data
  - StreamingAccuracy, StreamingPrecisionRecallF1, StreamingConfusionMatrix
  - RollingAccuracy
  - StreamingPipeline
  - Numerical stability (NaNs, zero-variance, single-class)

Run with:
    pytest tests/test_numcompute_stream.py -v
"""

import numpy as np
import pytest
import sys
import os

# Make sure package is importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.trees import DecisionTreeClassifier
from numcompute_stream.ensemble import (
    BaggingClassifier, RandomForestClassifier, EnsembleClassifier
)
from numcompute_stream.streaming import StreamTrainer, chunk_data
from numcompute_stream.metrics import (
    StreamingAccuracy, StreamingPrecisionRecallF1,
    StreamingConfusionMatrix, RollingAccuracy,
    accuracy, precision_recall_f1, confusion_matrix
)
from numcompute_stream.pipeline import StreamingPipeline, Pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_binary_data(n=200, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def make_multiclass_data(n=300, n_features=4, n_classes=3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (np.abs(X[:, 0]) * n_classes).astype(int) % n_classes
    return X, y


# ===========================================================================
# 1. DecisionTreeClassifier — Standard Functionality
# ===========================================================================

class TestDecisionTreeClassifier:

    def test_fit_predict_binary(self):
        X, y = make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_accuracy_above_chance(self):
        X, y = make_binary_data(n=300)
        clf = DecisionTreeClassifier(max_depth=5, random_state=0)
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert acc > 0.6, f"Expected accuracy > 0.6, got {acc:.4f}"

    def test_fit_predict_multiclass(self):
        X, y = make_multiclass_data()
        clf = DecisionTreeClassifier(max_depth=5, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape

    def test_entropy_criterion(self):
        X, y = make_binary_data()
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_classes_attribute(self):
        X, y = make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        assert set(clf.classes_) == {0, 1}

    def test_n_features_attribute(self):
        X, y = make_binary_data(n_features=6)
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        assert clf.n_features_in_ == 6

    def test_predict_proba_shape(self):
        X, y = make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(criterion="invalid")

    def test_invalid_max_depth_raises(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(max_depth=0)

    def test_shape_mismatch_raises(self):
        clf = DecisionTreeClassifier()
        X = np.ones((10, 3))
        y = np.ones(9)
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_predict_before_fit_raises(self):
        clf = DecisionTreeClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(np.ones((5, 3)))

    def test_wrong_feature_count_raises(self):
        X, y = make_binary_data(n_features=4)
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        with pytest.raises(ValueError):
            clf.predict(np.ones((5, 6)))


# ===========================================================================
# 2. DecisionTreeClassifier — Streaming
# ===========================================================================

class TestDecisionTreeStreaming:

    def test_partial_fit_two_chunks(self):
        X, y = make_binary_data(n=200)
        clf = DecisionTreeClassifier(max_depth=4, random_state=0)
        clf.partial_fit(X[:100], y[:100])
        clf.partial_fit(X[100:], y[100:])
        preds = clf.predict(X)
        assert preds.shape == (200,)

    def test_partial_fit_accumulates_data(self):
        X, y = make_binary_data(n=200)
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.partial_fit(X[:50], y[:50])
        assert clf._X_seen.shape[0] == 50
        clf.partial_fit(X[50:100], y[50:100])
        assert clf._X_seen.shape[0] == 100

    def test_partial_fit_with_nan(self):
        X, y = make_binary_data(n=100)
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        X_nan[5, 2] = np.nan
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.partial_fit(X_nan, y)  # Should not raise
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_partial_fit_single_class_chunk(self):
        """All-same-label chunk should not crash."""
        X, y = make_binary_data(n=100)
        clf = DecisionTreeClassifier(max_depth=3)
        clf.partial_fit(X[:50], np.zeros(50, dtype=int))
        clf.partial_fit(X[50:], y[50:])
        assert clf._root is not None

    def test_partial_fit_feature_mismatch_raises(self):
        clf = DecisionTreeClassifier(max_depth=3)
        clf.partial_fit(np.ones((10, 4)), np.zeros(10))
        with pytest.raises(ValueError):
            clf.partial_fit(np.ones((10, 5)), np.zeros(10))


# ===========================================================================
# 3. BaggingClassifier
# ===========================================================================

class TestBaggingClassifier:

    def test_fit_predict(self):
        X, y = make_binary_data(n=200)
        clf = BaggingClassifier(n_estimators=5, max_depth=3, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (200,)

    def test_partial_fit_streaming(self):
        X, y = make_binary_data(n=200)
        clf = BaggingClassifier(n_estimators=5, max_depth=3, random_state=0)
        for start in range(0, 200, 50):
            clf.partial_fit(X[start:start+50], y[start:start+50])
        preds = clf.predict(X)
        assert preds.shape == (200,)

    def test_score_reasonable(self):
        X, y = make_binary_data(n=400)
        clf = BaggingClassifier(n_estimators=10, max_depth=5, random_state=0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.6

    def test_estimators_count(self):
        X, y = make_binary_data()
        clf = BaggingClassifier(n_estimators=7, random_state=0)
        clf.fit(X, y)
        assert len(clf.estimators_) == 7

    def test_invalid_max_samples_raises(self):
        with pytest.raises(ValueError):
            BaggingClassifier(max_samples=1.5)


# ===========================================================================
# 4. RandomForestClassifier
# ===========================================================================

class TestRandomForestClassifier:

    def test_fit_predict(self):
        X, y = make_binary_data(n=200)
        rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert preds.shape == (200,)

    def test_partial_fit_multiple_chunks(self):
        X, y = make_binary_data(n=300)
        rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
        for X_c, y_c in chunk_data(X, y, chunk_size=100):
            rf.partial_fit(X_c, y_c)
        preds = rf.predict(X)
        assert preds.shape == (300,)

    def test_max_features_sqrt(self):
        X, y = make_binary_data(n_features=16)
        rf = RandomForestClassifier(n_estimators=5, max_features="sqrt", random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) >= 0.0

    def test_max_features_log2(self):
        X, y = make_binary_data(n_features=8)
        rf = RandomForestClassifier(n_estimators=5, max_features="log2", random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) >= 0.0

    def test_feature_importances_shape(self):
        X, y = make_binary_data(n_features=4)
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)
        imp = rf.feature_importances_()
        assert imp.shape == (4,)
        assert abs(imp.sum() - 1.0) < 1e-6

    def test_predict_proba_shape(self):
        X, y = make_binary_data(n=100)
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)
        proba = rf.predict_proba(X[:20])
        assert proba.shape[0] == 20

    def test_ensemble_classifier_alias(self):
        assert EnsembleClassifier is RandomForestClassifier

    def test_invalid_max_features_raises(self):
        X, y = make_binary_data()
        rf = RandomForestClassifier(max_features="bad_option")
        with pytest.raises(ValueError):
            rf.fit(X, y)


# ===========================================================================
# 5. StreamTrainer + chunk_data
# ===========================================================================

class TestStreamTrainer:

    def test_fit_chunk_logs(self):
        X, y = make_binary_data(n=200)
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        trainer = StreamTrainer(model, verbose=False)
        for X_c, y_c in chunk_data(X, y, chunk_size=50):
            trainer.fit_chunk(X_c, y_c)
        assert len(trainer.logs_) == 4

    def test_score_chunk(self):
        X, y = make_binary_data(n=200)
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        trainer = StreamTrainer(model, verbose=False)
        trainer.fit_chunk(X[:100], y[:100])
        score = trainer.score_chunk(X[100:], y[100:])
        assert 0.0 <= score <= 1.0

    def test_get_metric_history(self):
        X, y = make_binary_data(n=200)
        model = RandomForestClassifier(n_estimators=5, random_state=0)
        trainer = StreamTrainer(model, verbose=False)
        for X_c, y_c in chunk_data(X, y, chunk_size=50):
            trainer.fit_chunk(X_c, y_c)
        hist = trainer.get_metric_history("metric")
        assert hist.shape == (4,)

    def test_summary_keys(self):
        X, y = make_binary_data(n=100)
        model = DecisionTreeClassifier(random_state=0)
        trainer = StreamTrainer(model, verbose=False)
        trainer.fit_chunk(X[:50], y[:50])
        trainer.fit_chunk(X[50:], y[50:])
        s = trainer.summary()
        for key in ("n_chunks", "total_samples", "mean_metric",
                    "final_cumulative_metric", "total_time_s"):
            assert key in s

    def test_reset_clears_logs(self):
        X, y = make_binary_data(n=100)
        model = DecisionTreeClassifier(random_state=0)
        trainer = StreamTrainer(model, verbose=False)
        trainer.fit_chunk(X, y)
        trainer.reset()
        assert len(trainer.logs_) == 0
        assert trainer._chunk_idx == 0

    def test_chunk_data_sizes(self):
        X, y = make_binary_data(n=105)
        chunks = list(chunk_data(X, y, chunk_size=50))
        assert len(chunks) == 3
        assert chunks[0][0].shape[0] == 50
        assert chunks[2][0].shape[0] == 5

    def test_chunk_data_shuffle(self):
        X, y = make_binary_data(n=100)
        chunks1 = list(chunk_data(X, y, chunk_size=50, shuffle=False))
        chunks2 = list(chunk_data(X, y, chunk_size=50, shuffle=True, random_state=0))
        # Shuffled chunks should differ from unshuffled
        assert not np.array_equal(chunks1[0][0], chunks2[0][0])

    def test_chunk_data_invalid_chunk_size(self):
        X, y = make_binary_data(n=50)
        with pytest.raises(ValueError):
            list(chunk_data(X, y, chunk_size=0))


# ===========================================================================
# 6. Metrics
# ===========================================================================

class TestStreamingMetrics:

    def test_streaming_accuracy_update_result(self):
        acc = StreamingAccuracy()
        acc.update([0, 1, 1, 0], [0, 1, 0, 0])
        assert acc.result() == pytest.approx(3 / 4)

    def test_streaming_accuracy_multiple_updates(self):
        acc = StreamingAccuracy()
        acc.update([0, 1], [0, 1])
        acc.update([1, 0], [1, 1])
        assert acc.result() == pytest.approx(3 / 4)

    def test_streaming_accuracy_reset(self):
        acc = StreamingAccuracy()
        acc.update([0, 1, 1], [0, 1, 0])
        acc.reset()
        assert acc.result() == 0.0

    def test_streaming_accuracy_empty(self):
        acc = StreamingAccuracy()
        assert acc.result() == 0.0

    def test_precision_recall_f1_binary(self):
        prf = StreamingPrecisionRecallF1(average="binary", pos_label=1)
        prf.update([1, 0, 1, 1, 0], [1, 0, 0, 1, 1])
        r = prf.result()
        assert "precision" in r and "recall" in r and "f1" in r
        assert 0.0 <= r["precision"] <= 1.0

    def test_precision_recall_f1_macro(self):
        prf = StreamingPrecisionRecallF1(average="macro")
        prf.update([0, 1, 2, 0, 1, 2], [0, 1, 1, 0, 2, 2])
        r = prf.result()
        assert 0.0 <= r["f1"] <= 1.0

    def test_prf_reset(self):
        prf = StreamingPrecisionRecallF1()
        prf.update([0, 1], [0, 1])
        prf.reset()
        r = prf.result()
        assert r["f1"] == 0.0

    def test_confusion_matrix_binary(self):
        cm = StreamingConfusionMatrix(classes=[0, 1])
        cm.update([0, 1, 1, 0], [0, 1, 0, 1])
        mat = cm.result()
        assert mat.shape == (2, 2)
        assert mat.sum() == 4

    def test_confusion_matrix_accumulates(self):
        cm = StreamingConfusionMatrix(classes=[0, 1])
        cm.update([0, 1], [0, 1])
        cm.update([0, 1], [1, 0])
        mat = cm.result()
        assert mat.sum() == 4

    def test_rolling_accuracy_window(self):
        ra = RollingAccuracy(window_size=4)
        ra.update([0, 1, 1, 0, 1], [0, 1, 1, 0, 1])
        # Window keeps last 4: all correct
        assert ra.result() == pytest.approx(1.0)

    def test_rolling_accuracy_slides(self):
        ra = RollingAccuracy(window_size=3)
        ra.update([0, 0, 0], [1, 1, 1])  # 0 correct
        ra.update([1, 1, 1], [1, 1, 1])  # window now full of correct
        assert ra.result() == pytest.approx(1.0)

    def test_stateless_accuracy(self):
        assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75)

    def test_stateless_precision_recall_f1(self):
        r = precision_recall_f1([0, 1, 1], [0, 1, 0], average="binary")
        assert "f1" in r

    def test_stateless_confusion_matrix(self):
        mat = confusion_matrix([0, 1, 2], [0, 1, 1])
        assert mat.shape == (3, 3)


# ===========================================================================
# 7. StreamingPipeline
# ===========================================================================

class TestStreamingPipeline:

    def _make_scaler(self):
        """Simple streaming-compatible scaler for testing."""

        class MinMaxScaler:
            def __init__(self):
                self._min = None
                self._max = None

            def partial_fit(self, X):
                mn = np.nanmin(X, axis=0)
                mx = np.nanmax(X, axis=0)
                if self._min is None:
                    self._min = mn
                    self._max = mx
                else:
                    self._min = np.minimum(self._min, mn)
                    self._max = np.maximum(self._max, mx)

            def transform(self, X):
                rng = self._max - self._min
                rng[rng == 0] = 1.0
                return (X - self._min) / rng

        return MinMaxScaler()

    def test_pipeline_partial_fit_predict(self):
        X, y = make_binary_data(n=200)
        pipe = StreamingPipeline([
            ("scaler", self._make_scaler()),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=0))
        ])
        for X_c, y_c in chunk_data(X, y, chunk_size=50):
            pipe.partial_fit(X_c, y_c)
        preds = pipe.predict(X)
        assert preds.shape == (200,)

    def test_pipeline_score(self):
        X, y = make_binary_data(n=200)
        pipe = StreamingPipeline([
            ("scaler", self._make_scaler()),
            ("model", RandomForestClassifier(n_estimators=5, random_state=0))
        ])
        pipe.fit(X, y)
        assert 0.0 <= pipe.score(X, y) <= 1.0

    def test_pipeline_named_steps(self):
        scaler = self._make_scaler()
        model = DecisionTreeClassifier()
        pipe = StreamingPipeline([("scaler", scaler), ("model", model)])
        assert "scaler" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_pipeline_get_step(self):
        scaler = self._make_scaler()
        model = DecisionTreeClassifier()
        pipe = StreamingPipeline([("scaler", scaler), ("model", model)])
        assert pipe.get_step("model") is model

    def test_pipeline_alias(self):
        assert Pipeline is StreamingPipeline

    def test_pipeline_empty_steps_raises(self):
        with pytest.raises(ValueError):
            StreamingPipeline([])

    def test_pipeline_non_string_name_raises(self):
        with pytest.raises(TypeError):
            StreamingPipeline([(1, DecisionTreeClassifier())])

    def test_pipeline_missing_transform_raises(self):
        class BadTransformer:
            pass  # no .transform()
        with pytest.raises(TypeError):
            StreamingPipeline([
                ("bad", BadTransformer()),
                ("model", DecisionTreeClassifier())
            ])


# ===========================================================================
# 8. Numerical Stability Edge Cases
# ===========================================================================

class TestNumericalStability:

    def test_all_nan_column_handled(self):
        X = np.ones((20, 3))
        X[:, 1] = np.nan
        y = np.array([0, 1] * 10)
        clf = DecisionTreeClassifier(max_depth=2)
        clf.partial_fit(X, y)  # Should not crash
        preds = clf.predict(X)
        assert len(preds) == 20

    def test_zero_variance_feature(self):
        X = np.ones((30, 4))
        X[:, 2] = 99.0
        rng = np.random.default_rng(0)
        X[:, 0] = rng.standard_normal(30)
        y = (X[:, 0] > 0).astype(int)
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.5

    def test_single_sample_chunk(self):
        X, y = make_binary_data(n=50)
        clf = DecisionTreeClassifier(max_depth=3)
        clf.partial_fit(X[:1], y[:1])
        clf.partial_fit(X[1:], y[1:])
        preds = clf.predict(X)
        assert len(preds) == 50

    def test_large_chunk(self):
        X, y = make_binary_data(n=1000)
        rf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.6

    def test_streaming_accuracy_shape_mismatch_raises(self):
        acc = StreamingAccuracy()
        with pytest.raises(ValueError):
            acc.update([0, 1, 1], [0, 1])