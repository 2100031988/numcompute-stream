import re

import numpy as np

# The rank module provides functions for ranking data with tie handling and computing percentiles with various interpolation methods and These functions are implemented 
#  from scratch using only NumPy, without relying on external libraries.


# =========================
# RANKING (WITH TIES)
# =========================

def rank(data, method="average"):                           # It computes the ranks of the data while handling ties according to the specified method and the "method" parameter determines how ties
                                                            # are handled and can take values like "average", "dense", or "ordinal".

    """
    Compute ranks with tie handling.

    Parameters:
        data (array-like)
        method: 'average' | 'dense' | 'ordinal'

    Returns:
        np.ndarray of ranks
    """
    x = np.asarray(data)

    order = np.argsort(x, kind="mergesort")

    ranks = np.empty(len(x), dtype=float)

    if method == "ordinal":
        ranks[order] = np.arange(1, len(x) + 1)

    elif method == "average":
        sorted_x = x[order]
        i = 0
        while i < len(x):
            j = i
            while j < len(x) and sorted_x[j] == sorted_x[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2
            i = j

    elif method == "dense":
        sorted_x = x[order]
        rank_val = 1
        i = 0

        while i < len(x):
            j = i
            while j < len(x) and sorted_x[j] == sorted_x[i]:
                j += 1

            ranks[order[i:j]] = rank_val
            rank_val += 1
            i = j

    else:
        raise ValueError("method must be 'average', 'dense', or 'ordinal'")

    return ranks                                                                # it returns an array of ranks corresponding to the input data, with ties handled according to the specified method.


# =========================
# PERCENTILE FUNCTION
# =========================

def percentile(data, q, interpolation="linear"):                          # it returns the q-th percentile of the data using the specified interpolation method and the "interpolation" parameter determines 
                                                                          # how the percentile is computed when the desired percentile lies between two data points and can take values like "linear", "lower", "higher", or "midpoint".

    """
    Compute percentile with interpolation methods.

    Parameters:
        data (array-like)
        q (float): 0–100
        interpolation: 'linear' | 'lower' | 'higher' | 'midpoint'

    Returns:
        float
    """
    x = np.asarray(data)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return np.nan

    x = np.sort(x)

    if q < 0 or q > 100:
        raise ValueError("q must be between 0 and 100")

    pos = (q / 100) * (len(x) - 1)

    if interpolation == "lower":
        return x[int(np.floor(pos))]

    elif interpolation == "higher":
        return x[int(np.ceil(pos))]

    elif interpolation == "midpoint":
        lo = x[int(np.floor(pos))]
        hi = x[int(np.ceil(pos))]
        return (lo + hi) / 2

    elif interpolation == "linear":
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))

        if lo == hi:
            return x[lo]

        return x[lo] + (pos - lo) * (x[hi] - x[lo])

    else:
        raise ValueError("Invalid interpolation method")                        # returns the computed percentile value based on the specified interpolation method