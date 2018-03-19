'''SSCX functions related to working w/ hex'''
import numpy as np
import pandas as pd
from voxcell import VoxelData

from projectionizer.utils import XYZUVW

VOXEL_SIZE_UM = 10


def hexagon(hex_edge_len):
    '''make central column hexagon'''
    angles = np.arange(6 + 1) * ((2 * np.pi) / 6)
    points = hex_edge_len * np.transpose(np.array([np.cos(angles), np.sin(angles)]))
    return points


def get_virtual_fiber_locations(hex_edge_len, locations_path, apron_size=0.0):
    '''get locations in bounding box of central column'''
    points = hexagon(hex_edge_len)

    locations = pd.read_csv(locations_path)[['x', 'z']].values
    mean_locations = np.mean(locations, axis=0)

    min_xz = np.min(points, axis=0) + mean_locations - apron_size
    max_xz = np.max(points, axis=0) + mean_locations + apron_size

    idx = np.all((min_xz <= locations) & (locations <= max_xz), axis=1)
    return locations[idx]


def tiled_locations(voxel_size, hex_edge_len, locations_path):
    '''create grid spanning the bounding box of the central minicolum
    '''
    locations = get_virtual_fiber_locations(hex_edge_len, locations_path)

    min_x, min_z = np.min(locations, axis=0).astype(int)
    max_x, max_z = np.max(locations, axis=0).astype(int)

    x = np.arange(min_x, max_x + voxel_size, voxel_size)
    z = np.arange(min_z, max_z + voxel_size, voxel_size)

    grid = np.vstack(np.transpose(np.meshgrid(x, z)))

    return grid


def voxel_space(hex_edge_len, locations_path, max_height, voxel_size_um=VOXEL_SIZE_UM):
    '''returns VoxelData with the densities from `distmap`

    This is a 'stack' of (x == z == y == voxel_size) voxels stacked to
    the full y-height of the hexagon.  It can then be tiled across a whole
    space to get the desired density.

    Args:
        distmap: list of results of recipe_to_height_and_density()
        voxel_size(int): in um
    '''
    xz_extent = 1
    shape = (xz_extent, int(max_height // voxel_size_um), xz_extent)
    raw = np.zeros(shape=shape, dtype=np.int)

    tiles = tiled_locations(voxel_size_um,
                            hex_edge_len=hex_edge_len,
                            locations_path=locations_path)
    n_tile_x, n_tile_y = (tiles.max(axis=0) - tiles.min(axis=0)) / voxel_size_um
    raw = raw.repeat(n_tile_x, axis=0).repeat(n_tile_y, axis=2)
    return VoxelData(raw, [voxel_size_um] * 3, (tiles[:, 0].min(), 0, tiles[:, 1].min()))


def get_minicol_virtual_fibers(apron_size, hex_edge_len, locations_path):
    """returns Nx6 matrix: first 3 columns are XYZ pos of fibers, last 3 are direction vector"""

    fibers = set(tuple(loc)
                 for loc in get_virtual_fiber_locations(hex_edge_len=hex_edge_len,
                                                        locations_path=locations_path))
    extra_fibers = set(tuple(loc)
                       for loc in get_virtual_fiber_locations(apron_size=apron_size,
                                                              locations_path=locations_path,
                                                              hex_edge_len=hex_edge_len))
    extra_fibers = extra_fibers - fibers

    def to_dataframe(points, is_apron):
        '''return fibers in a dataframe'''
        df = pd.DataFrame(columns=XYZUVW, dtype=np.float)
        if not points:
            return df
        df.x, df.z = zip(*points)
        df.v = 1.  # all direction vectors point straight up
        df['apron'] = is_apron
        df['apron'] = df['apron'].astype(bool)
        return df.fillna(0)

    return pd.concat((to_dataframe(fibers, False),
                      to_dataframe(extra_fibers, True)),
                     ignore_index=True, sort=True)
