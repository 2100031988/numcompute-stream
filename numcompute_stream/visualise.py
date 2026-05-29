"""
visualise.py
------------
Lightweight plotting module for NumCompute-Stream.

All functions use only matplotlib. Supports inline display and
saving to file.

Required functions (per spec):
    - plot_metric_over_time(metric_values, title, ylabel)
    - compare_models(metric1, metric2, labels)
    - plot_predictions_vs_ground_truth(y_true, y_pred)

Author: NumCompute-Stream
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path: str = None):
    """Show figure inline or save to file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    save_path : str or None — if given, save instead of showing
    """
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[visualise] Saved plot to '{save_path}'")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Required plots
# ---------------------------------------------------------------------------

def plot_metric_over_time(metric_values, title: str = "Metric Over Time",
                           ylabel: str = "Metric",
                           xlabel: str = "Chunk",
                           color: str = "#2196F3",
                           save_path: str = None,
                           figsize=(9, 4)):
    """Plot a metric (e.g. accuracy) across streaming chunks.

    Parameters
    ----------
    metric_values : array-like, shape (n_chunks,)
        Metric value per chunk.
    title : str, default='Metric Over Time'
    ylabel : str, default='Metric'
    xlabel : str, default='Chunk'
    color : str, default='#2196F3'
    save_path : str or None — path to save figure (e.g. 'acc.png')
    figsize : tuple, default=(9, 4)

    Examples
    --------
    >>> plot_metric_over_time(trainer.get_metric_history(), title="Accuracy")
    """
    metric_values = np.array(metric_values, dtype=float)
    chunks = np.arange(len(metric_values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(chunks, metric_values, color=color, linewidth=2, marker="o",
            markersize=4, label=ylabel)
    ax.fill_between(chunks, metric_values, alpha=0.15, color=color)

    # Rolling mean overlay
    if len(metric_values) >= 5:
        window = min(5, len(metric_values))
        rolling = np.convolve(metric_values,
                              np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(metric_values)), rolling,
                linestyle="--", color="orange", linewidth=1.5,
                label=f"Rolling mean (w={window})")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    _save_or_show(fig, save_path)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   title: str = "Model Comparison",
                   ylabel: str = "Metric",
                   xlabel: str = "Chunk",
                   colors=("#2196F3", "#E91E63"),
                   save_path: str = None,
                   figsize=(9, 4)):
    """Compare two models on their streaming metric histories.

    Parameters
    ----------
    metric1 : array-like, shape (n_chunks,)
        Metric values for model 1.
    metric2 : array-like, shape (n_chunks,)
        Metric values for model 2.
    labels : tuple of str, default=('Model 1', 'Model 2')
    title : str, default='Model Comparison'
    ylabel : str, default='Metric'
    xlabel : str, default='Chunk'
    colors : tuple of str
    save_path : str or None
    figsize : tuple, default=(9, 4)

    Examples
    --------
    >>> compare_models(tree_acc, rf_acc, labels=("Single Tree", "Random Forest"))
    """
    m1 = np.array(metric1, dtype=float)
    m2 = np.array(metric2, dtype=float)
    n = max(len(m1), len(m2))
    chunks = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.arange(len(m1)), m1, color=colors[0], linewidth=2,
            marker="o", markersize=4, label=labels[0])
    ax.fill_between(np.arange(len(m1)), m1, alpha=0.1, color=colors[0])

    ax.plot(np.arange(len(m2)), m2, color=colors[1], linewidth=2,
            marker="s", markersize=4, label=labels[1])
    ax.fill_between(np.arange(len(m2)), m2, alpha=0.1, color=colors[1])

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    _save_or_show(fig, save_path)


def plot_predictions_vs_ground_truth(y_true, y_pred,
                                      title: str = "Predictions vs Ground Truth",
                                      save_path: str = None,
                                      max_samples: int = 100,
                                      figsize=(10, 4)):
    """Visualise predictions vs actual labels for the latest chunk.

    Shows a side-by-side scatter or strip plot of true vs predicted labels.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)
    title : str
    save_path : str or None
    max_samples : int, default=100 — cap displayed samples for clarity
    figsize : tuple, default=(10, 4)

    Examples
    --------
    >>> plot_predictions_vs_ground_truth(y_test, model.predict(X_test))
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = min(len(y_true), max_samples)
    idx = np.arange(n)
    yt = y_true[:n]
    yp = y_pred[:n]

    correct = yt == yp

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(idx[correct], yt[correct], color="#4CAF50", s=30,
               label="Correct", zorder=3, alpha=0.8)
    ax.scatter(idx[~correct], yt[~correct], color="#F44336", s=30,
               marker="x", label="Wrong (true)", zorder=3, alpha=0.9)
    ax.scatter(idx[~correct], yp[~correct], color="#FF9800", s=30,
               marker="^", label="Wrong (pred)", zorder=3, alpha=0.9)

    acc = float(np.mean(correct))
    ax.set_title(f"{title}  |  Accuracy={acc:.3f}", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Sample index", fontsize=11)
    ax.set_ylabel("Class label", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Additional useful plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes=None,
                           title: str = "Confusion Matrix",
                           cmap: str = "Blues",
                           save_path: str = None,
                           figsize=(6, 5)):
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    classes : array-like or None — class labels for axes
    title : str
    cmap : str, default='Blues'
    save_path : str or None
    figsize : tuple

    Examples
    --------
    >>> plot_confusion_matrix(y_test, preds, classes=[0, 1, 2])
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))

    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label", title=title)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_memory_over_time(logs: list,
                           title: str = "Memory Footprint Over Chunks",
                           save_path: str = None,
                           figsize=(9, 4)):
    """Plot model memory usage per chunk from StreamTrainer logs.

    Parameters
    ----------
    logs : list of dict — from StreamTrainer.logs_
    title : str
    save_path : str or None
    figsize : tuple
    """
    chunks = [r["chunk"] for r in logs]
    mem_kb = [r.get("memory_bytes", 0) / 1024 for r in logs]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(chunks, mem_kb, color="#9C27B0", alpha=0.75, width=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Chunk", fontsize=11)
    ax.set_ylabel("Memory (KB)", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_feature_importances(importances: np.ndarray,
                              feature_names=None,
                              title: str = "Feature Importances",
                              save_path: str = None,
                              figsize=(9, 4)):
    """Bar chart of feature importances from RandomForest.

    Parameters
    ----------
    importances : np.ndarray, shape (n_features,)
    feature_names : list of str or None
    title : str
    save_path : str or None
    figsize : tuple
    """
    importances = np.array(importances)
    n = len(importances)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n)]

    order = np.argsort(importances)[::-1]
    sorted_imp = importances[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(n), sorted_imp, color="#FF5722", alpha=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Importance", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_or_show(fig, save_path)