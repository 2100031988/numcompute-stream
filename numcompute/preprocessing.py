import numpy as np

# the preprocessing module provides basic data preprocessing tools like scaling, encoding, and imputation. Moreover, we have the same classes in the preprocessing module as in 
# scikit-learn but they are implemented from scratch using only NumPy. 


# =========================
# StandardScaler
# =========================

class StandardScaler:                               # It standardizes the data using Z-score so features like "mean" and "standard deviation" are calculated and used to scale the data.
                                                    # Moreover, the default value of mean is "0" and the default value of standard deviation is "1".

    """Z-score standardization"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)



# =========================
# MinMaxScaler
# =========================

class MinMaxScaler:                                 # It scales the data to a specified range typically [0, 1]. The "minimum" and "maximum" values of the features are calculated and used to scale the data accordingly. 
                                                    # The default range is [0, 1], but it can be customized by providing a different "feature_range" parameter during initialization.

    """Scale to a given range"""

    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.range = feature_range

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        denom = self.max_ - self.min_
        denom = np.where(denom == 0, 1.0, denom)

        X_std = (X - self.min_) / denom
        a, b = self.range
        return X_std * (b - a) + a

    def fit_transform(self, X):
        return self.fit(X).transform(X)



# =========================
# OneHotEncoder
# =========================

class OneHotEncoder:                                                # it converts categorical variables into binary vectors (0s and 1s) where each category is represented by a separate binary feature. 
                                                                    # The presence of a category is indicated by a "1" in the corresponding feature while the absence is indicated by a "0".

    """Convert categorical variables to one-hot encoding"""

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=object)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)

        encoded = [
            (X[:, i:i+1] == cats).astype(float)
            for i, cats in enumerate(self.categories_)
        ]

        return np.hstack(encoded)

    def fit_transform(self, X):
        return self.fit(X).transform(X)



# =========================
# SimpleImputer
# =========================

class SimpleImputer:                                        # It replaces missing values (NaN) in the dataset with a specified strategy like mean or constant value. Moreover, this simple impute function can be used to handle 
                                                            # missing data in numerical features by filling in the gaps with appropriate values based on the chosen strategy.

    """Replace NaN values"""

    def __init__(self, strategy="mean", fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)

        elif self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], self.fill_value)

        else:
            raise ValueError("strategy must be 'mean' or 'constant'")

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_out = X.copy()

        mask = np.isnan(X_out)
        X_out[mask] = self.statistics_[np.where(mask)[1]]

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)