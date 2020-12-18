"""
Provides utility functions.
"""


import numpy as np


def is_subvector(v1, v2):
    """
    Checks weather vector v1 is sub-vector of v2.

    Parameters
    ----------
    v1 : ndarray
        Vector to be checked if is contained in vector v2.
    v2 : ndarray
        Vector to be checked if contains vector v1.

    Returns
    -------
    bool
        True if v1 is sub-vector of v2, false otherwise.

    Examples
    --------

    >>> x1 = np.ones(3)
    >>> x2 = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    >>> is_subvector(x1, x2)
    True

    >>> x1 = np.ones(5)
    >>> x2 = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    >>> is_subvector(x1, x2)
    False

    """
    for k in range(len(v2) - len(v1) + 1):
        if np.array_equal(v2[k:k+len(v1)], v1):
            return True
    return False
