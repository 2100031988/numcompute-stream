"""
pipeline.py
-----------
Streaming-compatible Pipeline for NumCompute-Stream.

Chains transformers and a final estimator, supporting both
batch .fit() and incremental .partial_fit() for streaming data.

Author: NumCompute-Stream
"""

import numpy as np


class StreamingPipeline:
    """Chain of transformers followed by a final estimator.

    All steps must expose .partial_fit() or .fit() + .transform()
    (for transformers), and .partial_fit() + .predict() (for the
    final estimator).

    Parameters
    ----------
    steps : list of (name, object) tuples
        All but the last step must be transformers.
        The last step must be an estimator.

    Examples
    --------
    >>> from numcompute_stream.ensemble import RandomForestClassifier
    >>> from numcompute.preprocessing import StandardScaler
    >>> pipe = StreamingPipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('model', RandomForestClassifier())
    ... ])
    >>> pipe.partial_fit(X_chunk, y_chunk)
    >>> preds = pipe.predict(X_test)
    """

    def __init__(self, steps: list):

        if not steps:
            raise ValueError("steps must be a non-empty list.")
        
        self._validate_steps(steps)
        self.steps = steps

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_steps(steps):
        for i, (name, obj) in enumerate(steps):

            if not isinstance(name, str):
                raise TypeError(
                    f"Step name at index {i} must be a string, got {type(name)}"
                )
            
            if obj is None:
                raise ValueError(f"Step '{name}' is None.")

            if i < len(steps) - 1:
                if not hasattr(obj, "transform"):

                    raise TypeError(
                        f"Step '{name}' (index {i}) must have a .transform() method "
                        f"since it is not the final step."
                    )

            else:
                if not hasattr(obj, "predict"):
                    raise TypeError(
                        f"Final step '{name}' must have a .predict() method."
                    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def named_steps(self) -> dict:
        """Dict mapping step names to step objects."""
        return dict(self.steps)

    @property
    def _transformers(self):
        """All steps except the last."""
        return self.steps[:-1]

    @property
    def _estimator(self):
        """The final estimator step (name, obj)."""
        return self.steps[-1]

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit all transformers and transform X."""
        X_out = X

        for name, transformer in self._transformers:
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(X_out)

            elif hasattr(transformer, "fit"):
                transformer.fit(X_out)
            X_out = transformer.transform(X_out)

        return X_out

    def _transform_only(self, X: np.ndarray) -> np.ndarray:
        """Apply already-fitted transformers to X."""
        X_out = X

        for name, transformer in self._transformers:
            if not hasattr(transformer, "transform"):
                raise RuntimeError(
                    f"Transformer '{name}' has no .transform() method."
                )
            X_out = transformer.transform(X_out)

        return X_out


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StreamingPipeline":
        """Fit all steps on the full dataset.

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
        X_t = self._fit_transform(X, y)

        name, estimator = self._estimator

        if hasattr(estimator, "fit"):
            estimator.fit(X_t, y)
        elif hasattr(estimator, "partial_fit"):
            estimator.partial_fit(X_t, y)
        else:
            raise RuntimeError(
                f"Final estimator '{name}' has neither .fit() nor .partial_fit()."
            )
        
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    classes=None) -> "StreamingPipeline":
        """Incrementally update all steps with a new data chunk.

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

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
            )

        X_t = self._fit_transform(X, y)

        name, estimator = self._estimator

        if hasattr(estimator, "partial_fit"):
            estimator.partial_fit(X_t, y, classes=classes)
        elif hasattr(estimator, "fit"):
            estimator.fit(X_t, y)
        else:
            raise RuntimeError(
                f"Final estimator '{name}' has neither .partial_fit() nor .fit()."
            )
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform X and predict with final estimator.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """

        X = np.atleast_2d(np.array(X, dtype=float))
        X_t = self._transform_only(X)
        _, estimator = self._estimator

        return estimator.predict(X_t)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Transform X and return class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """

        X = np.atleast_2d(np.array(X, dtype=float))
        X_t = self._transform_only(X)
        _, estimator = self._estimator

        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                f"Final estimator '{self._estimator[0]}' has no .predict_proba()."
            )
        
        return estimator.predict_proba(X_t)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Predict and return accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """

        preds = self.predict(X)
        return float(np.mean(preds == np.array(y)))

    def get_step(self, name: str):
        """Retrieve a step by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        object : the step object
        """
        
        return self.named_steps[name]

    def __repr__(self):
        steps_str = " -> ".join(
            f"{name}({obj.__class__.__name__})" for name, obj in self.steps
        )

        return f"StreamingPipeline([{steps_str}])"


# Alias for compatibility with spec
Pipeline = StreamingPipeline