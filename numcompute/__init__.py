# description of the package

"""
NumCompute: A lightweight NumPy-based machine learning framework.

This package provides:
- Data IO utilities
- Preprocessing tools (scalers, encoders, imputers)
- Statistical functions
- Sorting and search algorithms
- Ranking and percentile tools
- ML evaluation metrics
- Numerical optimization (gradients, Jacobians)
- Pipeline abstraction system
"""

# this file is used as the main entry point for the package so we import all the key components here. Moerover, users can directly import it using any speicifc name of ther choice. 
# For example, import numcompute as np and then they can find mean of the data using np.mean().


from .io import load_csv        # read csv files

from .preprocessing import (        # preprocessing tools
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer
)

from .stats import (        # basic numarical operations
    mean,
    median,
    std,
    min,
    max,
    quantile,
    histogram
)

from .sort_search import (       # sorting and searching algorithms 
    stable_sort,
    multi_key_sort,
    topk,
    binary_search,
    quickselect
)

from .rank import rank, percentile  # use for ranking values and calculating percentiles

from .metrics import (      # basic evaluation metrics for machine learning models
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    mse,
    roc_curve,
    auc
)

from .optim import grad, jacobian   # numerical optimization tools for calculating gradients and Jacobians

from .pipeline import Pipeline, FeatureUnion, Transformer, Estimator    # basic pipeline abstraction system for building machine learning workflows

__all__ = [
    "load_csv",

    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",

    "mean",
    "median",
    "std",
    "min",
    "max",
    "quantile",
    "histogram",

    "stable_sort",
    "multi_key_sort",
    "topk",
    "binary_search",
    "quickselect",

    "rank",
    "percentile",

    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "mse",
    "roc_curve",
    "auc",

    "grad",
    "jacobian",

    "Pipeline",
    "FeatureUnion",
    "Transformer",
    "Estimator"
]