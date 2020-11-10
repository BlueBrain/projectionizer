import numpy as np
import pandas as pd
from voxcell.nexus.voxelbrain import Atlas
from nose.tools import ok_, eq_
from numpy.testing import assert_allclose, assert_equal, assert_array_equal

from voxcell import VoxelData
from projectionizer import sscx
from projectionizer.step_1_assign import assign_synapse_fiber
from projectionizer.utils import choice

from utils import TEST_DATA_DIR

from mocks import (
    create_candidates,
    create_synapse_counts,
    create_virtual_fibers
)


RAT_LAYERS = [('L6', 700),
              ('L5', 525),
              ('L4', 189),
              ('L3', 352),
              ('L2', 148),
              ('L1', 164),
              ]


def test_assign_synapse_fiber():
    np.random.seed(37)
    candidates = create_candidates()
    virtual_fibers = create_virtual_fibers()
    ret = assign_synapse_fiber(candidates, virtual_fibers, sigma=1)
    ok_(ret.equals(pd.DataFrame({'sgid': [0, 2, 1]})))


def test_column_layers():
    layer_starts, layer_thickness = sscx.column_layers(RAT_LAYERS)
    eq_(layer_starts, {'L6': 0, 'L4': 1225, 'L5': 700, 'L2': 1766, 'L3': 1414, 'L1': 1914})
    eq_(layer_thickness, {'L6': 700, 'L4': 189, 'L5': 525, 'L2': 148, 'L3': 352, 'L1': 164})


def test_recipe_to_height_and_density():
    profile = [[0.05, 0.01], [0.15, 0.02], [0.25, 0.03], [0.35, 0.04], [0.45, 0.04],
               [0.55, 0.04], [0.65, 0.03], [0.75, 0.02], [0.85, 0.01], [0.95, 0.01]]

    res = sscx.recipe_to_height_and_density(RAT_LAYERS, 'L4', 0, 'L3', 0.5, profile)
    assert_allclose(res,
                    [(1225.0, 0.01),
                     (1261.5, 0.02),
                     (1298.0, 0.03),
                     (1334.5, 0.04),
                     (1371.0, 0.04),
                     (1407.5, 0.04),
                     (1444.0, 0.03),
                     (1480.5, 0.02),
                     (1517.0, 0.01),
                     (1553.5, 0.01),
                     (1590.0, 0.01)])


def test_relative_recipe_to_height_and_density():
    profile = [[0.05, 0.01], [0.15, 0.02], [0.25, 0.03], [0.35, 0.04], [0.45, 0.04],
               [0.55, 0.04], [0.65, 0.03], [0.75, 0.02], [0.85, 0.01], [0.95, 0.01]]

    layers = [[6, 0],
              [5, 0],
              [4, 0],
              [3, 0],
              [2, 0],
              [1, 0]]


    layer4 = 4
    layer3 = 3
    res = sscx.recipe_to_relative_height_and_density(layers, layer4, 0, layer3, 0.5, profile)

    assert_allclose(res,
                    [[2.0, 0.01],
                     [2.15, 0.02],
                     [2.3, 0.03],
                     [2.45, 0.04],
                     [2.6, 0.04],
                     [2.75, 0.04],
                     [2.9, 0.03],
                     [3.05, 0.02],
                     [3.2, 0.01],
                     [3.35, 0.01],
                     [3.5, 0.01]])


def test_relative_distance_layer():
    layer_ph = np.array([[[[100, 200], [110, 200]],
                          [[110, 210], [120, 230]]]], dtype=float)
    distance = VoxelData(np.array([[[150, 146], [np.nan, 164]]], dtype=float), (1,1,1))

    res = sscx.relative_distance_layer(distance, layer_ph)

    assert_allclose(res.raw,
                     [[[0.5, 0.4],
                      [np.nan, 0.4]]])


def test_get_fiber_directions():
    # rotations for the positions in orientation.nrrd are:
    # [0,...] : 90deg x-axiswise
    # [1,...] : 90deg z-axiswise
    # [2,...] : 180deg x-axiswise
    fiber_pos = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')

    res = sscx.get_fiber_directions(fiber_pos, atlas)

    assert_allclose(res,
                    [(0, 0, 1),
                     (-1, 0, 0),
                     (0, -1, 0)], atol=1e-15)


def test_get_l5_l34_border_voxel_indices():
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')
    ret = sscx.get_l5_l34_border_voxel_indices(atlas, ['S1FL'])

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
    ret = sscx.mask_layers_in_regions(atlas, ['L3'], ['S1FL'])
    assert_array_equal(ret, mask)

    mask = brain_regions.raw == 1121
    mask |= brain_regions.raw == 1122
    mask |= brain_regions.raw == 1127
    mask |= brain_regions.raw == 1128
    ret = sscx.mask_layers_in_regions(atlas, ['L3', 'L4'], ['S1FL', 'S1HL'])
    assert_array_equal(ret, mask)


def test_mask_layer_6_bottom():
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')
    distance = atlas.load_data('[PH]y')

    # [PH]y is deliberately constructed from braint_regions so that the bottom of
    # S1J;L6 bottom has a distance of 0
    mask = brain_regions.raw == 1136
    mask &= distance.raw == 0

    ret = sscx.mask_layer_6_bottom(atlas, ['S1J'])
    assert_array_equal(ret, mask)


def test_recipe_to_relative_heights_per_layer():
    # [PH]6 is created from [PH]y. It's equally thick throughout the L6:
    # min limit is 0, max limit is the highest point of L6 in [PH]y.nrrd
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')
    distance = atlas.load_data('[PH]y')
    layers = [(6, None)]

    ret = sscx.recipe_to_relative_heights_per_layer(distance, atlas, layers)

    assert_array_equal(np.isfinite(ret.raw), atlas.get_region_mask('L6', attr='acronym').raw)
    assert_equal(True, np.all(ret.raw[np.isfinite(ret.raw)] >= 0.0))
    assert_equal(True, np.all(ret.raw[np.isfinite(ret.raw)] <= 1.0))

def test_ray_tracing():
    atlas = Atlas.open(TEST_DATA_DIR)
    brain_regions = atlas.load_data('brain_regions')
    distance = atlas.load_data('[PH]y')

    mask = brain_regions.raw == 1136
    mask &= distance.raw == 0

    ind_zero = np.transpose(np.where(mask))
    fiber_pos = distance.indices_to_positions(ind_zero) + np.array([0, 1000, 0])
    fiber_dir = np.zeros(fiber_pos.shape) + np.array([0, 1, 0])
    ret = sscx.ray_tracing(atlas, mask, fiber_pos, fiber_dir)
    ind_fibers = distance.positions_to_indices(ret[list('xyz')].values)

    assert_equal(len(ind_fibers), len(ind_zero))
    assert_array_equal(ind_fibers, ind_zero)
