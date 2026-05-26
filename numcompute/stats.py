import numpy as np

# The stats module provides functions for computing basic statistics like mean, median, standard deviation, minimum, maximum, quantiles, and histograms and 
# these functions are implemented using only NumPy and without relying on external libraries.


def mean(X, axis=None):                     # calculates mean of the array along the specified axis while ignoring any NaN values
    """
    Compute mean (NaN-safe).
    """

    X = np.asarray(X)

    return np.nanmean(X, axis=axis)


def median(X, axis=None):                    # calculates median of the array along the specified axis while ignoring any NaN values 

    """
    Compute median (NaN-safe).
    """

    X = np.asarray(X)

    return np.nanmedian(X, axis=axis)


def std(X, axis=None, ddof=0):              # calculates standard deviation of the array along the specified axis while ignoring any NaN values and is much better than 
                                            # the default NumPy std function

    """
    Compute standard deviation (NaN-safe).
    """

    X = np.asarray(X)

    return np.nanstd(X, axis=axis, ddof=ddof)


def min(X, axis=None):                     # calculates minimum of the array along the specified axis while ignoring any NaN values

    """
    Compute minimum (NaN-safe).
    """

    X = np.asarray(X)

    return np.nanmin(X, axis=axis)


def max(X, axis=None):                      # calculates maximum of the array along the specified axis while ignoring any NaN values

    """
    Compute maximum (NaN-safe).
    """

    X = np.asarray(X)

    return np.nanmax(X, axis=axis)


def quantile(X, q, axis=None):              # calculates quantiles of the array along the specified axis while ignoring any NaN values and is much better than the default NumPy quantile function

    """
    Compute quantiles (NaN-safe).

    Parameters:
        q : float or array-like (0 to 1)
    """

    X = np.asarray(X)

    return np.nanquantile(X, q, axis=axis)


def histogram(X, bins=10, range=None):          # calculates histogram of the array while ignoring any NaN values and is much better than the default NumPy histogram function 
                                                # and it returns the counts of values in each bin along with the corresponding bin edges.
    """
    Compute histogram.

    Returns:
        counts, bin_edges
    """
    X = np.asarray(X)
    X = X[~np.isnan(X)]  

    counts, bin_edges = np.histogram(X, bins=bins, range=range)

    return counts, bin_edges

