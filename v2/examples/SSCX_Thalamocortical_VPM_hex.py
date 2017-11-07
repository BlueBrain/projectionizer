'''Create projections for hexagonal circuit'''
import numpy as np
import pandas as pd
from voxcell import VoxelData

from examples.mini_col_locations import (get_virtual_fiber_locations,
                                         tiled_locations)
from projectionizer import sscx

VOXEL_SIZE_UM = 10
CLOSEST_COUNT = 25
EXCLUSION = 60


def voxel_space():
    '''returns VoxelData with the densities from `distmap`

    This is a 'stack' of (x == z == y == voxel_size) voxels stacked to
    the full y-height of the hexagon.  It can then be tiled across a whole
    space to get the desired density.

    Args:
        distmap: list of results of recipe_to_height_and_density()
        voxel_size(int): in um
    '''
    voxel_size = VOXEL_SIZE_UM

    xz_extent = 1
    shape = (xz_extent, int(sscx.LAYER_BOUNDARIES[-1] // voxel_size), xz_extent)
    raw = np.zeros(shape=shape, dtype=np.int)

    tiles = tiled_locations(VOXEL_SIZE_UM)
    n_tile_x, n_tile_y = (tiles.max(axis=0) -
                          tiles.min(axis=0)) / VOXEL_SIZE_UM
    raw = raw.repeat(n_tile_x, axis=0).repeat(n_tile_y, axis=2)
    return VoxelData(raw, [VOXEL_SIZE_UM] * 3, (tiles[:, 0].min(), 0, tiles[:, 1].min()))


def get_minicol_virtual_fibers():
    """returns Nx6 matrix: first 3 columns are XYZ pos of fibers, last 3 are direction vector"""
    apron_size = 50.

    fibers = set(tuple(loc) for loc in get_virtual_fiber_locations())
    extra_fibers = set(tuple(loc) for loc in get_virtual_fiber_locations(apron_size)) - fibers

    def to_dataframe(points, outsider):
        df = pd.DataFrame(columns=list('xyzuvw') + ['apron'])
        df.x, df.z = zip(*points)
        df.v = 1  # all direction vectors point straight up
        df.apron = outsider
        return df.fillna(0)

    return pd.concat((to_dataframe(fibers, False), to_dataframe(extra_fibers, True)))
