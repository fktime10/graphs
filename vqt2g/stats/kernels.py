"""Kernels for EMD"""

# Adapted from GRAN
# https://github.com/lrjconan/GRAN

import numpy as np
import pyemd

from scipy.linalg import toeplitz


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel with squared distance in exponential term replaced by EMD

    Args:
      x, y: 1D pmf of two distributions with the same support
      sigma: standard deviation (Default value = 1.0)
      x:
      y:
      distance_scaling:  (Default value = 1.0)

    Returns:

    """
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    """Gaussian kernel"""
    support_size = max(len(x), len(y))
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    """Gaussian total variation (TV) kernel"""
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))
