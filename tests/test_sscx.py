import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas

from projectionizer import sscx
from projectionizer.step_1_assign import assign_synapse_fiber

from mocks import create_candidates, create_virtual_fibers
from utils import TEST_DATA_DIR


def test_assign_synapse_fiber():
    np.random.seed(37)
    candidates = create_candidates()
    virtual_fibers = create_virtual_fibers()
    ret = assign_synapse_fiber(candidates, virtual_fibers, sigma=1)
    assert ret.equals(pd.DataFrame({"sgid": [0, 2, 1]}))


def test_relative_recipe_to_height_and_density():
    profile = [
        [0.05, 0.01],
        [0.15, 0.02],
        [0.25, 0.03],
        [0.35, 0.04],
        [0.45, 0.04],
        [0.55, 0.04],
        [0.65, 0.03],
        [0.75, 0.02],
        [0.85, 0.01],
        [0.95, 0.01],
    ]

    layers = [*np.arange(6, 0, -1)]

    layer4 = 4
    layer3 = 3
    res = sscx.recipe_to_relative_height_and_density(layers, layer4, 0, layer3, 0.5, profile)

    assert_allclose(
        res,
        [
            [2.0, 0.01],
            [2.15, 0.02],
            [2.3, 0.03],
            [2.45, 0.04],
            [2.6, 0.04],
            [2.75, 0.04],
            [2.9, 0.03],
            [3.05, 0.02],
            [3.2, 0.01],
            [3.35, 0.01],
            [3.5, 0.01],
        ],
    )


def test_relative_distance_layer():
    layer_ph = np.array(
        [
            [
                [[100, 200], [110, 200]],
                [[110, 210], [120, 230]],
            ]
        ],
        dtype=float,
    )
    distance = VoxelData(np.array([[[150, 146], [np.nan, 164]]], dtype=float), (1, 1, 1))

    res = sscx.relative_distance_layer(distance, layer_ph)

    assert_allclose(
        res.raw,
        [
            [
                [0.5, 0.4],
                [np.nan, 0.4],
            ]
        ],
    )


def test_recipe_to_relative_heights_per_layer():
    # [PH]6 is created from [PH]y. It's equally thick throughout the L6:
    # min limit is 0, max limit is the highest point of L6 in [PH]y.nrrd
    atlas = Atlas.open(TEST_DATA_DIR)
    distance = atlas.load_data("[PH]y")
    layers = ["L6"]

    ret = sscx.recipe_to_relative_heights_per_layer(distance, atlas, layers)

    assert_array_equal(np.isfinite(ret.raw), atlas.get_region_mask("L6", attr="acronym").raw)
    assert np.all(ret.raw[np.isfinite(ret.raw)] >= 0.0)
    assert np.all(ret.raw[np.isfinite(ret.raw)] <= 1.0)
