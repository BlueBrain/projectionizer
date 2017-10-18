#!/usr/bin/env python
import logging
import os
import sys
from functools import partial
from itertools import islice

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from voxcell import VoxelData

from decorators import pandas_cache, simple_cache, timeit
from mini_col_locations import (get_virtual_fiber_locations, hexagon,
                                tiled_locations)
from projectionizer import nrnwriter, projection, sscx, utils

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


def build_voxel_synapse_count(distmap, voxel_size):
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
            raw[0, bottom:top, 0] = int(voxel_size ** 3 * bottom_density)

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


TARGET_REMOVE = 1.6 / 2.6 * 0.73 / 0.66  # =~ 0.6806526806526807


# def find_cutoff_means_all_mtypes(synapses, mtypes):
#     '''For each mtype, find its unique cutoff_mean

#     Args:
#         circuit(bluepy.v2 circuit): circuit to work with
#         synapses(DataFrame): must have columns 'mtype', 'tgid', 'sgid'
#         target_remove(float): TARGET_REMOVE
#         bin_counts(int): bin count for the histogram

#     Returns:
#         dict(mtype -> cutoff_mean)

#     From thalamocortical_ps_s2f.py
#         compute cutoff by inverse interpolation of target fraction on cumulative syncount
#         distribution approximation: assumes hard cutoff, i.e. does not account for moments beyond
#         mean.  Should be OK if dist is fairly symmetrical.
#     '''
#     gb = synapses.groupby(['mtype', 'tgid', 'sgid']).size()

#     return {mtype: find_cutoff_mean(gb[mtype].value_counts(sort=False), config['target_remove'])
#             for mtype in mtypes}


def _find_cutoff_means(synapses, target_remove=TARGET_REMOVE):
    '''For each mtype, find its unique cutoff_mean

    Args:
        circuit(bluepy.v2 circuit): circuit to work with
        synapses(DataFrame): must have columns 'mtype', 'tgid', 'sgid'
        target_remove(float): TARGET_REMOVE
        bin_counts(int): bin count for the histogram

    Returns:
        dict(mtype -> cutoff_mean)

    From thalamocortical_ps_s2f.py
        compute cutoff by inverse interpolation of target fraction on cumulative syncount
        distribution approximation: assumes hard cutoff, i.e. does not account for moments beyond
        mean.  Should be OK if dist is fairly symmetrical.
    '''

    gb = synapses.groupby(['mtype', 'tgid', 'sgid']).size()
    mtypes = synapses.mtype.unique()
    return {mtype: find_cutoff_mean_per_mtype(gb[mtype].value_counts(sort=False), target_remove)
            for mtype in mtypes}


def find_cutoff_mean_per_mtype(value_count, target_remove):
    n_synapse_per_bin = np.array([value * count for value, count in value_count.iteritems()],
                                 dtype=float)
    x = np.cumsum(n_synapse_per_bin) / np.sum(n_synapse_per_bin)
    return np.interp(target_remove, xp=x, fp=value_count.index)


def prune_synapses_by_target_pathway(synapses, cutoff_var=1.0, parallelize=False):
    '''Based on the frequency of mtypes, and the synapses/connection frequency, probabilistically
    remove *connections* (ie: groups of synapses in a (sgid, tgid) pair
    '''
    cutoff_means = _find_cutoff_means(synapses)

    print(cutoff_means)

    def prune_based_on_cutoff(df):
        return np.random.random() < norm.cdf(len(df), cutoff_means[df['mtype'].iloc[0]], cutoff_var)

    if parallelize:
        import dask.dataframe as dd
        df = dd.from_pandas(synapses, npartitions=16)
        pathways = df.groupby(['sgid', 'tgid'])

        def prune_df(df):
            '''return dataframe if it satisfy cutoff probability'''
            if prune_based_on_cutoff(df):
                return df
            else:
                return None
        keep_syn = pathways.apply(prune_df, meta=synapses).compute()
        keep_syn.reset_index(drop=True, inplace=True)
    else:
        pathways = synapses.groupby(['sgid', 'tgid'])
        keep_syn = pathways.filter(prune_based_on_cutoff)

    return keep_syn


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

    distances = cdist(xz, fiber_locations)

    # get CLOSEST_COUNT closest virtual_fibers
    partition = np.argpartition(distances, CLOSEST_COUNT, axis=1)[:, :CLOSEST_COUNT]
    row_idx = np.arange(len(distances))[np.newaxis]
    distances = distances[row_idx.T, partition]

    # create mask for fibers that satisfy the EXCLUSION distance
    closest_fibers = fiber_locations[partition]
    closest_fibers = np.rollaxis(closest_fibers, 1)
    closest_fibers = np.abs(closest_fibers - xz) < (EXCLUSION, EXCLUSION)
    mask = np.all(closest_fibers, axis=2).T

    # want to choose the 'best' one based on a normal distribution based on distance
    distances[np.invert(mask)] = 1000  # make columns outside of exclusion unlikely to be pick
    distances = norm.pdf(distances, 0, SIGMA)

    # manually sample one of them: would use np.random.choice, but that's only the 1d case
    cum_distances = np.cumsum(distances, axis=1)
    cum_distances = cum_distances / np.sum(distances, axis=1, keepdims=True)
    rand_cutoff = np.random.random((len(cum_distances), 1))
    idx = np.argmax(rand_cutoff < cum_distances, axis=1)
    mini_cols = partition[row_idx, idx][0]

    mini_cols[np.invert(np.any(mask, axis=1))] = -1
    synapses['sgid'] = mini_cols

    return synapses


def calc_density(synapses, mult=1.):
    min_xyz, max_xyz = np.array([250, 0, 350]), np.array([550, 164, 700])
    distmap = [sscx.recipe_to_height_and_density('4', 0, '3', 0.5, y_distmap_3_4, mult=2.6),
               sscx.recipe_to_height_and_density('6', 0.85, '5', 0.6, y_distmap_5_6, mult=2.6),
               ]

    for dist in distmap:
        for (bottom, density), (top, _) in zip(dist[:-1], dist[1:]):
            min_xyz[1], max_xyz[1] = bottom, top
            volume = np.prod(max_xyz - min_xyz)
            in_bb = ((min_xyz[0] < synapses['x']) & (synapses['x'] < max_xyz[0]) &
                     (min_xyz[1] < synapses['y']) & (synapses['y'] < max_xyz[1]) &
                     (min_xyz[2] < synapses['z']) & (synapses['z'] < max_xyz[2]))
            count = float(np.count_nonzero(in_bb))
            got_density = count / volume
            print('Wanted: %.4f got: %.4f got_density/density: %.4f' %
                  (density, got_density, got_density / density))


@timeit('Pick Segments')
@simple_cache
def sample_synapses(tile_locations, circuit, voxel_size, map_):
    distmap = [sscx.recipe_to_height_and_density('4', 0, '3', 0.5, y_distmap_3_4, mult=2.6),
               sscx.recipe_to_height_and_density('6', 0.85, '5', 0.6, y_distmap_5_6, mult=2.6), ]

    voxel_synapse_count = build_voxel_synapse_count(distmap, voxel_size=voxel_size)
    synapses = pick_synapses(
        tile_locations, voxel_synapse_count, circuit, map_=map_)
    remove_cols = projection.START_COLS + projection.END_COLS
    synapses.drop(remove_cols, axis=1, inplace=True)
    return synapses


@timeit('Assign virtual fibers')
@simple_cache
def assign_synapses(synapses, map_):
    virtual_fiber_locations = get_virtual_fiber_locations()
    synapses.rename(columns={'gid': 'tgid'}, inplace=True)
    split = np.array_split(synapses, 100)
    asvf = partial(assign_synapse_virtual_fibers, fiber_locations=virtual_fiber_locations)
    split = map_(asvf, split)
    synapses = pd.concat(split)
    return synapses


@timeit('Prune attempt to get right distribution')
def prune(synapses, circuit, parallelize):
    synapses = synapses.join(circuit.cells.get(properties='mtype'), on='tgid')
    synapses.mtype.cat.remove_unused_categories(inplace=True)
    keep_syn = prune_synapses_by_target_pathway(synapses, parallelize)
    keep_syn.rename(columns={'Section.ID': 'section_id',
                             'Segment.ID': 'segment_id',
                             'segment_length': 'location',
                             }, inplace=True)
    keep_syn.columns = map(str, keep_syn.columns)
    return keep_syn


@timeit('write nrn.h5')
def write(synapses, output):
    nrn_path = os.path.join(output, 'nrn.h5')
    gb = synapses.groupby('tgid')
    nrnwriter.write_synapses(nrn_path, iter(gb), sscx.create_synapse_data)


def create_projections(output, circuit, parallelize):
    if parallelize:
        try:
            from dask.distributed import Client
        except ImportError:
            sys.exit("Need to install 'pip install 'dask[distributed]'")

        client = Client()

        def map_(func, it):
            res = client.map(func, it)
            return client.gather(res)
    else:
        map_ = map

    voxel_size = VOXEL_SIZE_UM
    tile_locations = tiled_locations(voxel_size=voxel_size)

    tile_locations = list(islice(tile_locations, 20))

    synapses = sample_synapses(tile_locations, circuit, voxel_size=voxel_size, map_=map_)
    assigned_synapses = assign_synapses(synapses, map_=map_)
    remaining_synapses = prune(assigned_synapses, circuit, parallelize)
    write(remaining_synapses, output)
