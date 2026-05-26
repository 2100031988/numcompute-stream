from platform import machine

import numpy as np

# pipeline.py aims to combine multiple steps into a workflow and that includes collecting raw data then preprocessing it, then applying some transformations and finally fitting a model to the data
# where we do some transformations on the data to get desired output and then we can use the output to make predictions or evaluate the model performance.


# =========================
# BASE CLASSES (Protocol)
# =========================

class Transformer:                                      # a base class for data transformers that includes methods for "fitting" and "transforming data" where we can use it to create custom 
                                                        # transformers for data preprocessing, feature engineering, etc.
                                                        
    """
    Base Transformer interface.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Estimator:                                         # a base class for estimators (models) that learns from the data and make predictions using fit and predict methods where we can use it to create custom 
                                                         # models for regression, classification, etc.
    """
    Base Estimator interface 
    """

    def fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError


# =========================
# PIPELINE
# =========================

class Pipeline:                                                 # a system that applies multiple steps sequentially and here the each steps input becomes the output of the previous step and it is used to build 
                                                                # machine learning workflows where we can combine data preprocessing, feature engineering and model fitting into a single pipeline.

    """
    Sequential pipeline for transformers and estimators.
    """

    def __init__(self, steps):
        """
        steps: list of (name, transformer/estimator)
        """
        self.steps = steps

    def fit(self, X, y=None):
        for _, step in self.steps:
            if hasattr(step, "fit"):
                if "y" in step.fit.__code__.co_varnames:
                    step.fit(X, y)
                else:
                    step.fit(X)
            if hasattr(step, "transform"):
                X = step.transform(X)
        return self

    def transform(self, X):
        for _, step in self.steps:
            if hasattr(step, "transform"):
                X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# =========================
# FEATURE UNION
# =========================

class FeatureUnion:                                                   # It is a system that applies multiple transformations in parallel and combine their outputs into a single dataset where we can use it to apply 
                                                                      # multiple feature extraction or transformation steps to the same input data and then concatenate their outputs for further processing or modeling.

    """
    Apply multiple transformers in parallel and concatenate outputs.
    """

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        for _, t in self.transformers:
            t.fit(X, y)
        return self

    def transform(self, X):
        outputs = [t.transform(X) for _, t in self.transformers]
        return np.hstack(outputs)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)