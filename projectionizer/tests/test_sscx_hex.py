import os
from voxcell import VoxelData
import numpy as np
from numpy.testing import assert_array_equal
from projectionizer import sscx_hex, utils


BOUNDING_BOX = np.array([[110, 400], [579.99, 799.99]])
BOUNDING_BOX_APRON = [BOUNDING_BOX[0] - 10,  BOUNDING_BOX[1] + 10]
TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
LOCATIONS_PATH = os.path.join(TEMPLATES, 'rat_fibers.csv')


def test_get_virtual_fiber_locations():
    vf = sscx_hex.get_virtual_fiber_locations(bounding_box=BOUNDING_BOX,
                                              locations_path=LOCATIONS_PATH)
    assert len(vf) == 414

    vf = sscx_hex.get_virtual_fiber_locations(bounding_box=BOUNDING_BOX_APRON,
                                              locations_path=LOCATIONS_PATH)
    assert len(vf) > 414


def test_get_minicol_virtual_fibers():
    # Create voxel data that resembles BOUNDING_BOX
    offset = [-230, 0, 0]
    height = np.full((120, 10, 120), np.nan)
    height[34:81, :, 40:80] = 1
    height = VoxelData(height, [10, 10, 10], offset=offset)

    # Create a region mask
    mask = np.isfinite(height.raw)

    df = sscx_hex.get_minicol_virtual_fibers(apron_bounding_box=[],
                                             height=height,
                                             region_mask=mask,
                                             locations_path=LOCATIONS_PATH)
    assert len(df) == 414
    assert len(df[df.apron]) == 0

    # Create voxel data that resembles BOUNDING_BOX_APRON
    height = height.raw
    height[33:82, :, 39:81] = 1
    height = VoxelData(height, [10, 10, 10], offset)

    df = sscx_hex.get_minicol_virtual_fibers(apron_bounding_box=BOUNDING_BOX_APRON,
                                             height=height,
                                             region_mask=mask,
                                             locations_path=LOCATIONS_PATH)
    assert len(df) > 414
    assert len(df[df.apron]) > 0

    assert df.apron.dtype == bool
    for c in utils.XYZUVW:
        assert df[c].dtype == float


def test_get_mask_bounding_box():
    # Create voxel data that resembles BOUNDING_BOX
    offset = [-230, 0, 0]
    mask = np.full((120, 10, 120), np.nan)
    mask[34:81, :, 40:80] = 1
    height = VoxelData(mask, [10, 10, 10], offset=offset)

    mask = sscx_hex.get_mask_bounding_box(height, mask, BOUNDING_BOX)

    assert_array_equal(np.isfinite(height.raw), mask)
