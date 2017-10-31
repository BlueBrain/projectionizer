from numpy.testing import assert_equal
import math

from voxcell import VoxelData

from nose.tools import ok_, eq_, raises
from numpy.testing import assert_allclose
import numpy as np

from projectionizer import sscx


def test_calc_distances():
    locations = np.array([[0,  0,  0],
                          [10,  0,  0],
                          [10, 10, 10],
                          [10, 10,  0]], dtype=np.float)
    virtual_fibers = np.array([[0.,  0.,  0.,  1.,  0.,  0.],
                               [0.,  0.,  0.,  0.,  1.,  0.],
                               [0.,  0.,  0.,  0.,  0.,  1.]])
    ret = sscx.calc_distances(locations, virtual_fibers)
    eq_(ret.shape, (4, 3))
    sqrt2 = math.sqrt(2)
    expected = np.array([[0., 0., 0.],  # line passes through origin
                         [0., 10., 10.], # x line passes through x=10, so 0 distance there, 10 for y & z lines
                         [10. * sqrt2, 10. * sqrt2, 10. * sqrt2], # symmetric distances to point at 10, 10, 10
                         [10., 10., 10. * sqrt2],  # point on xy plane, so 10 distance from x & y lines, right angle to z
                         ])

    assert_allclose(expected, ret)


def _create_synapse_counts():
    synapse_counts = VoxelData(np.zeros((5, 5, 5)), [10] * 3, (10, 10, 10))
    idx = np.array([[2, 2, 2],  # middle of cube: xyz = (35, 35, 35)
                    [2, 2, 3]]) # xyz = (35, 35, 45)
    synapse_counts.raw[tuple(idx.T)] = 1
    return synapse_counts


def _create_virtual_fibers():
    virtual_fibers = np.array([[30.,  30.,  30.,  1.,  0.,  0.],  # (30, 30, 30), along x axis
                               [30.,  30.,  30.,  0.,  1.,  0.],  # (30, 30, 30), along y axis
                               [30.,  30.,  30.,  0.,  0.,  1.]]) # (30, 30, 30), along z axis
    return virtual_fibers


def test_get_voxelized_fiber_distances():
    synapse_counts = _create_synapse_counts()
    virtual_fibers = _create_virtual_fibers()
    closest_count = 5
    exclusion = 10
    ret = sscx.get_voxelized_fiber_distances(synapse_counts,
                                             virtual_fibers,
                                             closest_count=closest_count,
                                             exclusion=exclusion)
    assert_equal(ret[(2, 2, 2)], np.array([0, 1, 2], dtype=int))  # all are equal distance
    assert_equal(ret[(2, 2, 3)], np.array([2, 1, 0], dtype=int))  # 2 is closest, other two are outside of extent


def test_assign_synapse_fiber():
    np.random.seed(37)
    locations = np.array([[35., 35., 35.],
                          [35., 35., 45.]])
    synapse_counts = _create_synapse_counts()
    virtual_fibers = _create_virtual_fibers()
    voxelized_fiber_distances = sscx.get_voxelized_fiber_distances(synapse_counts, virtual_fibers)
    ret = sscx.assign_synapse_fiber(locations,
                                    synapse_counts,
                                    virtual_fibers,
                                    voxelized_fiber_distances)
    eq_(ret, [2, 1])


def test_mask_far_fibers():
    fibers = np.array([[[0, 0], [2, 2], [3, 3], [4, 4]],
                       [[-1, 0], [2.5, 2.0], [3, 3], [4, 2]]])
    mask = sscx.mask_far_fibers(fibers, [2, 2], (2, 2))
    assert_equal(mask,
                 [[False,  True, True, False],
                  [False,  True,  True, False]])


def test_choice():
    np.random.seed(0)
    indices = sscx.choice(np.array([[1., 2, 3, 4],
                                    [0, 0, 1, 0],
                                    [6, 5, 4, 0]]))
    assert_equal(indices,
                 [2, 2, 1])
