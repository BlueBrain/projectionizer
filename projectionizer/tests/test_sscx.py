import numpy as np
import pandas as pd
from nose.tools import ok_, eq_
from numpy.testing import assert_allclose, assert_equal

from voxcell import VoxelData
from projectionizer import sscx
from projectionizer.step_1_assign import assign_synapse_fiber
from projectionizer.utils import choice

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
