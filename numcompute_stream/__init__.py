"""
numcompute_stream
-----------------
A streaming, decision tree–based machine learning framework
built on top of NumCompute.

Modules
-------
trees       : DecisionTreeClassifier
ensemble    : BaggingClassifier, RandomForestClassifier, EnsembleClassifier
streaming   : StreamTrainer, chunk_data
metrics     : StreamingAccuracy, StreamingPrecisionRecallF1,
              StreamingConfusionMatrix, RollingAccuracy,
              accuracy, precision_recall_f1, confusion_matrix
pipeline    : StreamingPipeline (alias: Pipeline)
visualise   : plot_metric_over_time, compare_models,
              plot_predictions_vs_ground_truth,
              plot_confusion_matrix, plot_memory_over_time,
              plot_feature_importances
"""

# <--------- Imports ----------->

from numcompute_stream.trees import DecisionTreeClassifier
from numcompute_stream.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    EnsembleClassifier,
)
from numcompute_stream.streaming import StreamTrainer, chunk_data
from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingPrecisionRecallF1,
    StreamingConfusionMatrix,
    RollingAccuracy,
    accuracy,
    precision_recall_f1,
    confusion_matrix,
)
from numcompute_stream.pipeline import StreamingPipeline, Pipeline
from numcompute_stream import visualise


# <--------- Package Metadata ----------->

__version__ = "1.0.0"
__author__ = "NumCompute-Stream"


# <--------- Public API ----------->

__all__ = [
    # Trees important for streaming
    "DecisionTreeClassifier",

    # Ensemble methods for streaming
    "BaggingClassifier",
    "RandomForestClassifier",
    "EnsembleClassifier",

    # Streaming utilities
    "StreamTrainer",
    "chunk_data",

    # Metrics for streaming evaluation
    "StreamingAccuracy",
    "StreamingPrecisionRecallF1",
    "StreamingConfusionMatrix",
    "RollingAccuracy",
    "accuracy",
    "precision_recall_f1",
    "confusion_matrix",

    # Pipeline utilities
    "StreamingPipeline",
    "Pipeline",

    # Visualise module for streaming metrics and results
    "visualise",
]