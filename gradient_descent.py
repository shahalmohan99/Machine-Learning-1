import numpy as np
from typing import Callable, Tuple


def gradient_descent(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                     df: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                     x0: np.ndarray, 
                     y0: np.ndarray, 
                     learning_rate: float, 
                     lr_decay: float, 
                     num_iters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find a local minimum of the function f(x, y) using gradient descent:
    Until the number of iteration is reached, decrease the current x and y points by the
    respective partial derivative times the learning_rate.
    In each iteration, record the current function value in the list f_list 
    and the current x and y points in the lists x_list and y_list.
    The function should return the lists x_list, y_list, f_list.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate: Learning rate
    :param lr_decay: Learning rate decay
    :param num_iters: number of iterations
    :return: x_list, y_list, f_list (lists of x, y, and f values over iterations). 
             The first element of the lists represents the initial point (and the function value at the initial point).
             The last element of the lists represents the final point (and the function value at the final point).
    """
    f_list = np.zeros(num_iters+1)
    x_list = np.zeros(num_iters+1)
    y_list = np.zeros(num_iters+1)

    # TODO: Implement the gradient descent algorithm with a decaying learning rate
    
    # Initialize starting point
    x = x0
    y = y0
    x_list[0] = x
    y_list[0] = y
    f_list[0] = f(x, y)

    # Perform gradient descent
    for t in range(1, num_iters + 1):
        grad_x, grad_y = df(x, y)  # Get gradient at current point
        lr = learning_rate * (lr_decay ** (t - 1))  # Apply learning rate decay

        # Update x and y
        x = x - lr * grad_x
        y = y - lr * grad_y

        # Store updated values
        x_list[t] = x
        y_list[t] = y
        f_list[t] = f(x, y)

    return x_list, y_list, f_list


def rastrigin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Implements the Rastrigin function (as specified in the assignment sheet)
    :param x: x-coordinate
    :param y: y-coordinate
    :return: Rastrigin function value
    """
    # TODO: Implement the Rastrigin function (as specified in the Assignment 1 sheet)
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))


def gradient_rastrigin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Implements partial derivatives of Rastrigin function w.r.t. x and y
    :param x: x-coordinate
    :param y: y-coordinate
    :return: Gradient of Rastrigin function
    """
    # TODO: Implement partial derivatives of the Rastrigin function w.r.t. x and y
    df_dx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    df_dy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)

    gradient = np.array([df_dx, df_dy])
    return gradient


def finite_difference_gradient_approx(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                                      x: np.ndarray, 
                                      y: np.ndarray, 
                                      h: float = 1e-5) -> np.ndarray:
    """
    Implement finite difference gradient approximation.
    :param f: Function to approximate the gradient of
    :param x: x-coordinate
    :param y: y-coordinate
    :param h: Step size
    :return: Approximated gradient
    """
    # TODO: Implement numerical approximation to the partial derivatives of 
    # the Rastrigin function w.r.t. x and y
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)

    approx_grad = np.array([df_dx, df_dy])
    return approx_grad