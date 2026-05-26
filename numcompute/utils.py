import numpy as np

# this modules contains various utility functions for distance calculations, activation functions, log-sum-exp trick for numerical stability, top-k selection, 
# and batching utilities and these functions are implemented using only NumPy.

# =========================
# DISTANCES
# =========================

def euclidean_distance(a, b):                                       # it computes the Euclidean distance between two vectors or batches "a" and "b" and uses NumPy's array operations 
                                                                    # to calculate the distance by taking the square root of the sum of squared differences between corresponding elements in the input arrays.

    """
    Euclidean distance between two batches of vectors.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def manhattan_distance(a, b):                                   # it computes the Manhattan distance between two vectors "a" and "b" and uses NumPy's array operations to efficiently 
                                                                # calculate the distance by taking the sum of absolute differences between corresponding elements in the input arrays.

    """
    L1 distance
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.sum(np.abs(a - b), axis=-1)


# =========================
# ACTIVATIONS
# =========================

def sigmoid(x):                                             # sigmoid activation function returns a value between 0 and 1 and is commonly used in binary classification problems.
    
    """
    Sigmoid activation function
    """
     
    x = np.asarray(x, dtype=float)      
    return 1 / (1 + np.exp(-x))


def relu(x):                                                # ReLU (Rectified Linear Unit) activation function returns the input value if it is positive and 0 otherwise 
                                                            # and is widely used in deep learning models.
    
    """ 
    ReLU activation function
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(0, x)


def softmax(x):                                             # softmax activation function converts a vector of raw scores into probabilities and is commonly used 
                                                            # in multi-class classification problems.

    """
    Softmax activation function
    """

    x = np.asarray(x, dtype=float)

    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shift)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# =========================
# LOG-SUM-EXP (STABILITY)
# =========================

def logsumexp(x, axis=None):                                # logsumexp returns the logarithm of the sum of exponentials of the input array "x" along the specified axis while ensuring numerical stability 
                                                            # by subtracting the maximum value before exponentiation to prevent overflow issues.

    """
    Stable log-sum-exp trick.
    """

    x = np.asarray(x, dtype=float)

    max_x = np.max(x, axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))


# =========================
# TOP-K HELPER
# =========================

def top_k(x, k, largest=True):                          # it retrieves the top k elements from the input array "x" based on the specified parameters and uses NumPy's "argpartition" function to 
                                                        # efficiently find the indices of the top k elements,    

    """
    Return top-k values and indices.
    """

    x = np.asarray(x)

    if largest:
        idx = np.argpartition(x, -k)[-k:]
    else:
        idx = np.argpartition(x, k)[:k]

    vals = x[idx]

    order = np.argsort(-vals if largest else vals)
    return vals[order], idx[order]


# =========================
# BATCHING UTILITIES
# =========================

def batch_iterator(X, batch_size=32, shuffle=True):             # it generates batches of data  with a specified batch size and an option to shuffle the data before creating batches and 
                                                                # returns batches of data as NumPy arrays that can be used for training machine learning models

    """
    Yield batches from dataset.
    """

    X = np.asarray(X)
    n = len(X)

    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, n, batch_size):                       # here for loop is used to go through the dataset step by step and create smaller batches instead of processing all data at once.
        batch_idx = indices[i:i + batch_size]
        yield X[batch_idx]