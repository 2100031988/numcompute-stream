import numpy as np


# =========================
# GRADIENT (FINITE DIFFERENCE)
# =========================

def grad(f, x, h=1e-5, method="central"):                       # we use gradient estimation to compute the vector of partial derivatives of a scalar function f at a given point x where we can choose between 
                                                                # "forward" and "central" difference methods for numerical approximation. Use case includes optimization problems, machine learning, etc.

    """
    Estimate gradient of scalar function f at x.

    Parameters:
        f (callable): function R^n → R
        x (array-like): point at which to evaluate gradient
        h (float): step size
        method: 'forward' or 'central'

    Returns:
        np.ndarray: gradient vector
    """

    x = np.asarray(x, dtype=float)
    grad_vec = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x_step = x.copy()

        if method == "forward":
            x_step[i] += h
            grad_vec[i] = (f(x_step) - f(x)) / h

        elif method == "central":
            x_step[i] += h
            f_plus = f(x_step)

            x_step[i] = x[i] - h
            f_minus = f(x_step)

            grad_vec[i] = (f_plus - f_minus) / (2 * h)

        else:
            raise ValueError("method must be 'forward' or 'central'")

    return grad_vec                                                        


# =========================
# JACOBIAN (VECTOR OUTPUT FUNCTION)
# =========================

def jacobian(F, x, h=1e-5, method="central"):                 # Jacobian estimation is defined as the matrix of all first-order partial derivatives of a vector-valued function F at a given 
                                                              # point x where we can choose between "forward" and "central" difference methods for numerical approximation.        
                                                  
    """
    Estimate Jacobian matrix of vector function F at x.

    Parameters:
        F (callable): R^n → R^m
        x (array-like): input point
        h (float): step size
        method: 'forward' or 'central'

    Returns:
        np.ndarray: Jacobian matrix (m × n)
    """
    x = np.asarray(x, dtype=float)

    f0 = np.asarray(F(x))
    m = f0.size
    n = x.size

    J = np.zeros((m, n), dtype=float)

    for i in range(n):
        x_step = x.copy()

        if method == "forward":
            x_step[i] += h
            diff = (np.asarray(F(x_step)) - f0) / h

        elif method == "central":
            x_step[i] += h
            f_plus = np.asarray(F(x_step))

            x_step[i] = x[i] - h
            f_minus = np.asarray(F(x_step))

            diff = (f_plus - f_minus) / (2 * h)

        else:
            raise ValueError("method must be 'forward' or 'central'")

        J[:, i] = diff

    return J                                                        # It is used in various applications such as gradietn-based optimization and particularly in neural networks like backpropagation where we need to 
                                                                    # compute the Jacobian of the loss function with respect to the model parameters.