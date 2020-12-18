import numpy as np

from smf import utils


def test_is_subvector_if_v1_is_subvector_of_v2():
    v1 = np.ones(3)
    v2 = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    assert utils.is_subvector(v1, v2)


def test_is_subvector_if_v1_is_not_subvector_of_v2():
    v1 = np.ones(5)
    v2 = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    assert not utils.is_subvector(v1, v2)
