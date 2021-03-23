import math

import numpy as np
import pandas as pd
from nose.tools import eq_, ok_, raises
from numpy.testing import assert_allclose, assert_equal
from pandas.testing import assert_frame_equal

from projectionizer import straight_fibers

from mocks import (
    create_candidates,
    create_synapse_counts,
    create_virtual_fibers
)


def test_calc_distances_vectorized():
    np.random.seed(37)
    candidates = create_candidates()
    virtual_fibers = create_virtual_fibers()
    ret = straight_fibers.calc_distances_vectorized(candidates, virtual_fibers)
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
    ret = straight_fibers.calc_distances(locations, virtual_fibers)
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


def test_calc_pathlength_to_fiber_start():
    locations = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [1, 1, 1]
                          ])
    sgid_fibers = np.repeat(np.array([[0, 0, 0, 1, 0, 0]]), 3, axis=0)

    ret = straight_fibers.calc_pathlength_to_fiber_start(locations, sgid_fibers)
    assert_allclose(ret, [0,
                          1,  # going directly along the x axis
                          1 + math.sqrt(2)]) # shortest distance x-axis, then from there to origin

    basis = 1. / np.linalg.norm([1, 1, 1])
    sgid_fibers = np.repeat(np.array([[0, 0, 0, basis, basis, basis]]), 3, axis=0)
    ret = straight_fibers.calc_pathlength_to_fiber_start(locations, sgid_fibers)
    assert_allclose(ret, [0,
                          1.39384685,
                          np.linalg.norm([1, 1, 1])]) # distance to origin


def test_closest_fibers_per_voxel():
    synapses = create_synapse_counts()
    virtual_fibers = create_virtual_fibers()

    assert_frame_equal(straight_fibers.closest_fibers_per_voxel(synapses, virtual_fibers, 3),
                       pd.DataFrame([[0, 1, 2, 2, 2],
                                     [2, 1, 2, 2, 3]],
                                    columns=[0, 1, 'i', 'j', 'k']),
                       check_dtype=False)
