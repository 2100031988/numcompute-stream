from shutil import which

import numpy as np


# The sort_search module provides functions for stable sorting, multi-key sorting, top-k selection, binary search, and quickselect and these 
# functions are implemented from scratch using only NumPy, without relying on external libraries.


# =========================
# Stable Sort
# =========================

def stable_sort(X, axis=-1):                                    # It performs a stable sort on the input array X along the specified axis using NumPy's sort function with the "stable" sorting algorithm.

    """
    Stable sort wrapper using NumPy.

    Parameters:
        X (array-like): input array
        axis (int): axis to sort along

    Returns:
        np.ndarray: sorted array
    """

    X = np.asarray(X)
    return np.sort(X, axis=axis, kind="stable")


# =========================
# Multi-key Sort
# =========================

def multi_key_sort(X, keys):                                    # it returns a sorted version of the input 2D array X based on multiple column keys specified in the "keys" parameter and the sorting is performed using NumPy's 
                                                                # np.lexsort function which allows for sorting by multiple keys in a stable manner.

    """
    Sort rows of a 2D array using multiple column keys.

    Parameters:
        X (ndarray): shape (n_samples, n_features)
        keys (list[int]): column priority order

    Returns:
        np.ndarray: sorted array
    """

    X = np.asarray(X)

    if X.ndim != 2:

        raise ValueError("multi key sort expects a 2D array")


    sort_indices = np.lexsort([X[:, k] for k in reversed(keys)])
    return X[sort_indices]


# =========================
# Top-K (Partial Sort)
# =========================

def topk(values, k, largest=True, return_indices=True):             # Top-k is one of the most common operations in data processing and machine learning where we want to find the top k largest or smallest elements from a dataset and 
                                                                    # this function efficiently retrieves the top k elements from the input array "values" based on the specified parameters.

    """
    Return top-k elements using np.argpartition.

    Parameters:
        values (array-like)
        k (int)
        largest (bool): True = largest k elements
        return_indices (bool)

    Returns:
        values or (values, indices)
    """

    values = np.asarray(values)

    if k <= 0:
        raise ValueError("k must be > 0")

    k = min(k, len(values))

    if largest:
        idx = np.argpartition(values, -k)[-k:]
    else:
        idx = np.argpartition(values, k)[:k]

    top_vals = values[idx]

    order = np.argsort(-top_vals if largest else top_vals)
    idx = idx[order]
    top_vals = top_vals[order]

    return (top_vals, idx) if return_indices else top_vals


# =========================
# Binary Search
# =========================

def binary_search(sorted_array, x):                             # it performs a binary search on a sorted array to find the index of a specified value "x" and it uses NumPy's "searchsorted" function to efficiently locate 
                                                                # the position where "x" would be inserted in the sorted array while maintaining the order.
    """
    Binary search using NumPy searchsorted.

    Parameters:
        sorted_array (array-like): must be sorted
        x: value to search

    Returns:
        (index, found)
    """

    arr = np.asarray(sorted_array)

    idx = np.searchsorted(arr, x)
    found = (idx < len(arr)) and (arr[idx] == x)

    return idx, bool(found)


# =========================
# Quickselect (Educational)
# =========================

def quickselect(arr, k):                                    # it finds the k-th smallest element in an unsorted array using the Quickselect algorithm which is an efficient selection algorithm and works by partitioning 
                                                            # the array around a pivot element and narrowing down the search space until the k-th smallest element is found.

    """
    Find k-th smallest element using partition method.

    Parameters:
        arr (array-like)
        k (int)

    Returns:
        value at kth position
    """

    arr = np.asarray(arr).copy()

    if k < 0 or k >= len(arr):
        raise ValueError("k index out of bounds")

    return np.partition(arr, k)[k]