import math

import numpy as np
import pandas as pd
from nose.tools import eq_, ok_, raises
from numpy.testing import assert_allclose, assert_equal

from projectionizer.fibers import *
from projectionizer.tests.mocks import (create_candidates,
                                        create_synapse_counts,
                                        create_virtual_fibers)


def test_calc_distances():
    np.random.seed(37)
    candidates = create_candidates()
    virtual_fibers = create_virtual_fibers()
    ret = calc_distances_vectorized(candidates, virtual_fibers)
    assert_allclose(ret, [[0.70710678,  0.70710678,  0.70710678],
                          [0.70710678,  0.70710678,  0.70710678],
                          [0.70710678,  0.70710678,  0.70710678]])


def test_calc_distances():
    locations = np.array([[0,  0,  0],
                          [10,  0,  0],
                          [10, 10, 10],
                          [10, 10,  0]], dtype=np.float)
    virtual_fibers = np.array([[0.,  0.,  0.,  1.,  0.,  0.],
                               [0.,  0.,  0.,  0.,  1.,  0.],
                               [0.,  0.,  0.,  0.,  0.,  1.]])
    ret = calc_distances(locations, virtual_fibers)
    eq_(ret.shape, (4, 3))
    sqrt2 = math.sqrt(2)
    expected = np.array([[0., 0., 0.],  # line passes through origin
                         [0., 10., 10.],  # x line passes through x=10, so 0 distance there, 10 for y & z lines
                         # symmetric distances to point at 10, 10, 10
                         [10. * sqrt2, 10. * sqrt2, 10. * sqrt2],
                         # point on xy plane, so 10 distance from x & y lines, right angle to z
                         [10., 10., 10. * sqrt2],
                         ])

    assert_allclose(expected, ret)


def test_closest_fibers_per_voxel():
    synapses = create_synapse_counts()
    fibers = create_virtual_fibers()

    ok_(closest_fibers_per_voxel(synapses, fibers, 3).equals(
        pd.DataFrame([[0,  1,  2,  2,  2],
                      [2,  1,  2,  2,  3]],
                     columns=[0, 1, 'i', 'j', 'k'])))
