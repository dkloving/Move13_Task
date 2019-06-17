import numpy as np
import cv2
from scipy.stats import linregress
from scipy.optimize import differential_evolution


def optimize(lines_lst, image_width, image_height, max_iterations=2000):
    """
    Tries to find the value of parameter `a` used in `arctan_unwarp` that best transforms all sets of points
    into straight lines.

    :param image_width:
        int, image width in pixels
    :param image_height:
        int, image height in pixels
    :param lines_lst:
        list of point sets. Each element should be an array of points, shaped as (n_points, 2).
    :param max_iterations:
        int, maximum number of iterations the optimizer should perform
    :return:
        optimal value of a
    """
    if not all([type(item) == np.ndarray for item in lines_lst]):
        raise TypeError("lines_lst must contain numpy arrays")
    if not all([item.shape[-1] == 2 for item in lines_lst]):
        raise ValueError("lines_lst items must all be of shape (n, 2)")
    # make function for optimizing
    optme = lambda x: _optimizable(lines_lst, x, image_width, image_height)
    optim_result = differential_evolution(optme, bounds=[(1e-8, 2)], maxiter=max_iterations, seed=42)
    return optim_result.x


def arctan_unwarp(points, a, image_width, image_height):
    """
    Transforms points to remove lens distortion according to an arctan model. Cannot produce the
     identity function, but asymptotically approaches it as `a` goes toward 0

    :param points:
        ndarray of image space points
    :param a:
        real-valued scalar, tunable parameter.
    :param image_width:
        int, image width in pixels
    :param image_height:
        int, image image_height in pixels
    :returns:
        array of points after transformation
    """
    x = points[..., 0]
    y = points[..., 1]
    # center
    x_center = x - image_width / 2
    y_center = y - image_height / 2
    # convert to polar coords
    r_polar, theta_polar = cv2.cartToPolar(x_center, y_center)
    # # make lens correction
    r_max = np.hypot(image_width, image_height) * 0.5
    r_polar = r_max * np.tan(r_polar * np.arctan(a) / r_max) / a
    # convert back to cartesian coords and un-center
    x_cart, y_cart = cv2.polarToCart(r_polar, theta_polar)
    x = np.squeeze(x_cart) + image_width / 2
    y = np.squeeze(y_cart) + image_height / 2
    # make array
    points_adjusted = np.stack([x, y], axis=-1)
    return points_adjusted


def _linear_fit(lines_lst):
    """
    Uses OLS to fit lines to each set of points, then returns the sum goodness of fit over all lines.

    :return:
        scalar representation of goodness of fit
    """
    r_values = []  # higher is better!
    for line in lines_lst:
        _, _, r_value, _, _ = linregress(line[:, 0], line[:, 1])
        r_values.append(r_value)
    return sum(r_values)


def _optimizable(lines_lst, a, image_width, image_height):
    """
    Function to be optimized in order to find a good lens unwarp parameter

    :return:
        real-valued scalar which should be minimized
    """
    transformed_points_lst = [arctan_unwarp(line_points, a, image_width, image_height) for line_points in lines_lst]
    score = -_linear_fit(transformed_points_lst)
    return score
