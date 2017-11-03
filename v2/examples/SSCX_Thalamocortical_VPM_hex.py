'''Create projections for hexagonal circuit'''
import logging
import os
import sys
from functools import partial
from itertools import islice

import numpy as np
import pandas as pd
import toolz
from scipy.spatial.distance import cdist
from scipy.stats import norm
from voxcell import VoxelData

from decorators import pandas_cache, timeit
from mini_col_locations import get_virtual_fiber_locations, tiled_locations
from projectionizer import projection, sscx, utils

L = logging.getLogger(__name__)

VOXEL_SIZE_UM = 10
BASE_CIRCUIT = '/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/merged_circuit/'


# from thalamocorticalProjectionRecipe.xml
# tuple of (percent of height of region, density in synapses / um^3
# the 'percent height' is the midpoint, so, ((0.05, 0.01), (0.15, 0.02), ...)
# means that the
# density 0.01 is used from 0.0 - 0.1,
y_distmap_3_4 = (
    (0.05, 0.01),
    (0.15, 0.02),
    (0.25, 0.03),
    (0.35, 0.04),
    (0.45, 0.04),
    (0.55, 0.04),
    (0.65, 0.03),
    (0.75, 0.02),
    (0.85, 0.01),
    (0.95, 0.01),
)
y_distmap_5_6 = (
    (0.05, 0.005),
    (0.15, 0.01),
    (0.25, 0.015),
    (0.35, 0.02),
    (0.45, 0.0225),
    (0.55, 0.025),
    (0.65, 0.0275),
    (0.75, 0.03),
    (0.85, 0.015),
    (0.95, 0.005),
)


def get_distmap():
    return [sscx.recipe_to_height_and_density('4', 0, '3', 0.5, y_distmap_3_4),
            sscx.recipe_to_height_and_density('6', 0.85, '5', 0.6, y_distmap_5_6), ]


def build_voxel_synapse_count(distmap, voxel_size, oversampling):
    '''returns VoxelData with the densities from `distmap`

    This is a 'stack' of (x == z == y == voxel_size) voxels stacked to
    the full y-height of the hexagon.  It can then be tiled across a whole
    space to get the desired density.

    Args:
        distmap: list of results of recipe_to_height_and_density()
        voxel_size(int): in um
    '''
    xz_extent = 1
    shape = (xz_extent, int(sscx.LAYER_BOUNDARIES[-1] // voxel_size), xz_extent)
    raw = np.zeros(shape=shape, dtype=np.int)

    for dist in distmap:
        for (bottom, bottom_density), (top, top_density) in zip(dist[:-1], dist[1:]):
            bottom = int(bottom // voxel_size)
            top = int(top // voxel_size)
            #density = np.interp(np.arange(bottom, top), [bottom, top], [bottom_density, top_density])
            assert xz_extent == 1, "Must change raw's dimensions if xz_extent changes"
            #raw[0, bottom:top, 0] = np.rint(voxel_size ** 3 * density).astype(np.int)
            raw[0, bottom:top, 0] = int(voxel_size ** 3 * bottom_density * oversampling)

    return VoxelData(raw, [voxel_size] * 3, (0, 0, 0))


def _pick_synapses(xz_loc, circuit, voxel_synapse_count):
    '''helper to pick synapses: for each xz_loc, get the synapses in that 'tower'

    Args:
        xz_loc(tuple float): the x/z location at which to find synapses in 'tower'
        circuit(bluepy.v2 circuit): circuit to work with
        voxel_synapse_count: 'tower' of synapse counts

    Must be in global scope so can be used in parallel
    '''
    voxel_size = voxel_synapse_count.voxel_dimensions[0]

    min_i = int(xz_loc[0] // voxel_size)
    max_i = min_i + voxel_synapse_count.raw.shape[0]

    min_k = int(xz_loc[1] // voxel_size)
    max_k = min_k + voxel_synapse_count.raw.shape[2]

    min_ijk = (min_i, 0, min_k)
    max_ijk = (max_i, voxel_synapse_count.raw.shape[1], max_k)

    # shift column to the correct xz_location
    voxel_synapse_count.offset = np.array((xz_loc[0], 0, xz_loc[1]))

    column_synapses = projection.pick_synapses(
        circuit, voxel_synapse_count, min_ijk, max_ijk, utils.segment_pref)
    return column_synapses


def pick_synapses(tile_locations, voxel_synapse_count, circuit, map_=map):
    '''parallelize picking synapses: each tile_location is run separately

    Args:
        tile_locations: iterable of tuples of float (x, z) tile_locations to apply the
        voxel_synapse_count to
        voxel_synapse_count: VoxelData where x & z are 1 unit, and y is as tall as the volume
        to sample
        circuit(bluepy.v2 circuit): circuit to work with
        map_=map: map function

    '''
    get_syns_func = partial(_pick_synapses,
                            voxel_synapse_count=voxel_synapse_count,
                            circuit=circuit)
    synapses = map_(get_syns_func, tile_locations)

    synapses = pd.concat(synapses)
    synapses.drop(list('ijk'), axis=1, inplace=True)

    return synapses


def first_partition(big_array, size, axis=1):
    """Returns a tuple of 2 elements:
        - an array containing the first partition of big_array
        - the indices of these elements"""
    indices = np.argpartition(big_array, size, axis=axis)[:, :size]
    row_idx = np.arange(len(big_array))[np.newaxis]
    big_array = big_array[row_idx.T, indices]
    return big_array, indices


def assign_synapse_virtual_fibers(synapses, fiber_locations):
    '''Each potential synapse needs to be assigned to a unique virtual fiber

    Args:
        synapses(pandas df): must have 'x', 'z' columns
        fiber_locations(numpy array(Nx2)): X/Z positions of the virtual fibers
    '''
    EXCLUSION = 60  # 3 times std?
    SIGMA = 20
    CLOSEST_COUNT = 25

    # TODO: do EXCLUSION first
    xz = synapses[['x', 'z']].values

    distances, indices = first_partition(cdist(xz, fiber_locations), CLOSEST_COUNT)

    mask = sscx.mask_far_fibers(fiber_locations[indices], xz, (EXCLUSION, EXCLUSION))

    # want to choose the 'best' one based on a normal distribution based on distance
    distances[np.invert(mask)] = 1000  # make columns outside of exclusion unlikely to be pick
    idx = sscx.choice(norm.pdf(distances, 0, SIGMA))
    row_idx = np.arange(len(indices))[np.newaxis]
    mini_cols = indices[row_idx, idx][0]
    mini_cols[np.invert(np.any(mask, axis=1))] = -1

    synapses['sgid'] = mini_cols

    return synapses


@timeit('Pick Segments')
def sample_synapses(tile_locations, circuit, voxel_size, map_, n_islice):
    # p = '/gpfs/bbp.cscs.ch/project/proj30/mgevaert/csThal/orig_v5_2p6_no_interp/segments.feather'
    # synapses = pd.read_feather(p)
    # return synapses.iloc[:n_islice]

    voxel_synapse_count = build_voxel_synapse_count(get_distmap(), voxel_size, oversampling=2.6)
    synapses = pick_synapses(
        tile_locations, voxel_synapse_count, circuit, map_=map_)
    remove_cols = utils.SEGMENT_START_COLS + utils.SEGMENT_END_COLS
    synapses.drop(remove_cols, axis=1, inplace=True)
    return synapses.iloc[:n_islice]


def get_minicol_virtual_fibers():
    "returns Nx6 matrix: first 3 columns are XYZ pos of fibers, last 3 are direction vector"

    virtual_fiber_locations = get_virtual_fiber_locations()
    virtual_fibers = np.zeros((len(virtual_fiber_locations), 6))
    virtual_fibers[:, 0] = virtual_fiber_locations[:, 0]  # X
    virtual_fibers[:, 2] = virtual_fiber_locations[:, 1]  # Z

    # all direction vectors point straight up
    virtual_fibers[:, 4] = 1

    return virtual_fibers


@timeit('Assign vector virtual fibers')
def assign_synapses_vector_fibers(synapses, map_):
    '''Assign virtual fibers from vectors'''
    synapses.rename(columns={'gid': 'tgid'}, inplace=True)

    virtual_fibers = get_minicol_virtual_fibers()

    xyz = synapses[list('xyz')].values
    min_ = np.min(xyz, axis=0)
    max_ = np.max(xyz, axis=0)

    voxel_size = VOXEL_SIZE_UM
    raw = (1 + max_ - min_) // voxel_size
    synapse_counts = VoxelData(np.zeros(shape=raw.astype(int)), [voxel_size] * 3, min_)
    idx = np.unique(synapse_counts.positions_to_indices(xyz), axis=0)
    synapse_counts.raw[tuple(idx.T)] = 1

    voxelized_fiber_distances = sscx.get_voxelized_fiber_distances(synapse_counts, virtual_fibers)

    asf = partial(sscx.assign_synapse_fiber,
                  synapse_counts=synapse_counts,
                  virtual_fibers=virtual_fibers,
                  voxelized_fiber_distances=voxelized_fiber_distances)

    CHUNK_SIZE = 100000
    it = toolz.itertoolz.partition_all(CHUNK_SIZE, xyz)
    fiber_id = map_(asf, it)
    fiber_id = np.hstack(fiber_id)
    synapses['sgid'] = fiber_id
    return synapses


@timeit('Assign virtual fibers')
def assign_synapses(synapses, map_):
    virtual_fiber_locations = get_virtual_fiber_locations()
    synapses.rename(columns={'gid': 'tgid'}, inplace=True)
    split = np.array_split(synapses, 100)
    assign_func = partial(assign_synapse_virtual_fibers, fiber_locations=virtual_fiber_locations)
    split = map_(assign_func, split)
    synapses = pd.concat(split)
    return synapses
