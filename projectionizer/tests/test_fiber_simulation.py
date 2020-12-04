import pandas as pd
import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_allclose,)

from voxcell.nexus.voxelbrain import Atlas
from projectionizer import fiber_simulation

from utils import TEST_DATA_DIR

XZ = list('xz')


def test_generate_kmeans_fibers():
    # coordinates for a square
    sqc = np.array([[0,0],
                    [0,2],
                    [2,2],
                    [2,0]])

    xv = np.array([1, 0])
    zv = np.array([0, 1])

    # creat
    xz = np.concatenate([sqc,
                         sqc + 10 * xv,
                         sqc + 10 * zv,
                         sqc + 10,
                         ]).astype(float)

    cells = pd.DataFrame(xz, columns=XZ)
    v_dir = 1.0
    y_level = 0.0
    n_fibers = 4

    fibers = fiber_simulation._generate_kmeans_fibers(cells, n_fibers, v_dir, y_level)

    assert_equal(True, np.all(fibers['v'] == v_dir))
    assert_equal(True, np.all(fibers['y'] == y_level))
    assert_equal(len(fibers), n_fibers)

    assert_array_equal(fibers.sort_values(XZ)[XZ].values, [[1, 1],
                                                           [1, 11],
                                                           [11, 1],
                                                           [11, 11]])


def test_get_fiber_directions():
    # rotations for the positions in orientation.nrrd are:
    # [0,...] : 90deg x-axiswise
    # [1,...] : 90deg z-axiswise
    # [2,...] : 180deg x-axiswise
    fiber_pos = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    atlas = Atlas.open(TEST_DATA_DIR)

    res = fiber_simulation.get_fiber_directions(fiber_pos, atlas)

    assert_allclose(res,
                    [(0, 0, 1),
                     (-1, 0, 0),
                     (0, -1, 0)], atol=1e-15)


def test_get_l5_l34_border_voxel_indices():
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')
    ret = fiber_simulation.get_l5_l34_border_voxel_indices(atlas, ['S1FL'])

    region_ids = brain_regions.raw[tuple(np.transpose(ret))]
    assert_array_equal([1123], np.unique(region_ids))

    neighbors = np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

    def check_neighbors(ind, layer_ids):
        # check that any of the `layer_ids` is found in the neighbors
        idx = tuple(np.transpose(ind + neighbors))
        return np.any(np.in1d(brain_regions.raw[idx], layer_ids))

    assert_equal(True, np.all([check_neighbors(i, [1121, 1122]) for i in ret]))


def test_mask_layers_in_regions():
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')

    mask = brain_regions.raw == 1121
    ret = fiber_simulation.mask_layers_in_regions(atlas, ['L3'], ['S1FL'])
    assert_array_equal(ret, mask)

    mask = brain_regions.raw == 1121
    mask |= brain_regions.raw == 1122
    mask |= brain_regions.raw == 1127
    mask |= brain_regions.raw == 1128
    ret = fiber_simulation.mask_layers_in_regions(atlas, ['L3', 'L4'], ['S1FL', 'S1HL'])
    assert_array_equal(ret, mask)


def test_mask_layer_6_bottom():
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')
    distance = atlas.load_data('[PH]y')

    # [PH]y is deliberately constructed from braint_regions so that the bottom of
    # S1J;L6 bottom has a distance of 0
    mask = brain_regions.raw == 1136
    mask &= distance.raw == 0

    ret = fiber_simulation.mask_layer_6_bottom(atlas, ['S1J'])
    assert_array_equal(ret, mask)


def test_ray_tracing():
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')
    distance = atlas.load_data('[PH]y')

    mask = brain_regions.raw == 1136
    mask &= distance.raw == 0

    ind_zero = np.transpose(np.where(mask))
    fiber_pos = distance.indices_to_positions(ind_zero) + np.array([0, 1000, 0])
    fiber_dir = np.zeros(fiber_pos.shape) + np.array([0, 1, 0])
    ret = fiber_simulation.ray_tracing(atlas, mask, fiber_pos, fiber_dir)
    ind_fibers = distance.positions_to_indices(ret[list('xyz')].values)

    assert_equal(len(ind_fibers), len(ind_zero))
    assert_array_equal(ind_fibers, ind_zero)
