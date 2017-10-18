#!/usr/bin/env python
'''projectionizer v2

General Overview:

1) Create a voxcell.VoxelData object w/ the synapse count expected
   per voxel.  EX: in the examples: build_voxel_synapse_count

2) Write a 'segment_pref' function, this should assign probabilities to each
   section found in a voxel, and allows one to filter which segments are used

3) If the voxcell.VoxelData object is 'dense', not very big, or the
    one can directly use pick_synapses() over the whole space
    (using minijk, maxijk).  If knowledge of the space allows one to speed up
    the process, one can write repeatedly call pick_synapses(), or read
    it's source, and perform similar calculations.

4) Assign synapses created about to a soma, ie: a 'sgid' (source GID.)
   For virtual projections, one will have to come up with a number scheme that
   doesn't conflict with other GIDS.  One can use assign_synapse_to_cell() if
   it applies here.

5) Write the synapse data to an nrn.h5 style file, using
    nrnwriter:py.write_synapses()

Note:  Notationally, when i, j, k are used, it means in 'voxel space', and when x, y, z
       are used, it means in 'world space'
'''

import itertools as it
import logging
import os

import numpy as np
import pandas as pd

from datetime import datetime

from bluepy.v2.enums import Section, Segment

import utils

IJK = utils.IJK
I, J, K = 0, 1, 2
L = logging.getLogger(__name__)

START_COLS = [Segment.X1, Segment.Y1, Segment.Z1, ]
END_COLS = [Segment.X2, Segment.Y2, Segment.Z2, ]
WANTED_COLS = ['gid', Section.ID, Segment.ID, 'segment_length', 'x', 'y', 'z',
               ] + START_COLS + END_COLS


def assign_synapse_to_cell_voxel(voxel_probability, cells, synapses, presyn_pref):
    '''Assign all synapses

    Args:
        voxel_probability(3D np.array): the probability of the voxels containing the
        cell which will innervate the synapse.  This 3D map is assumed to be centered
        on the synapse's location
        synapses(DataFrame): synapses that need to be assigned, has the `WANTED_COLS` columns
        cells(DataFrame of cells): must have 'i', 'j', 'k' columns corresponding to voxel indices
        presyn_pref(callable: (cells DataFrame, synapse) -> probability):

    Returns:
        gid of the chosen cell
    '''
    # pick voxel for each synapse, this voxel will be where the soma will be chosen from
    shape = np.array(voxel_probability.raw.shape)
    idx = np.random.choice(np.arange(np.prod(shape)), size=len(synapses),
                           p=np.ravel(voxel_probability.raw))
    voxel_positions = np.transpose(np.array(np.unravel_index(idx, shape)))
    offset = (voxel_probability.offset - voxel_probability.offset // 2).astype(np.int)
    synapses[['src_i', 'src_j', 'src_k']] = pd.DataFrame(voxel_positions + offset,
                                                         index=synapses.index,
                                                         dtype=np.int)
    failed = []
    ret = {}
    grouped_cells = cells.groupby(IJK)
    for voxel_pos, group_synapses in synapses.groupby(['src_i', 'src_j', 'src_k']):
        start = datetime.now()

        candidate_cells = grouped_cells.get_group(voxel_pos)
        for i, synapse in group_synapses.iterrows():
            prob_density = presyn_pref(candidate_cells, synapse)
            try:
                prob_density = utils.normalize_probability(prob_density)
                gid = np.random.choice(candidate_cells.index.values, p=prob_density)
            except utils.ErrorCloseToZero:
                failed.append(synapse['gid'])
                gid = -1

            ret[i] = int(synapse['gid']), int(synapse[Segment.ID]), gid

    ret = pd.DataFrame.from_dict(data=ret, orient='index', dtype=np.int)
    ret.columns = ['target_gid', Segment.ID, 'source_gid']

    L.debug('Failed: %s', len(failed))
    return ret


def assign_synapse_to_cell(cells, voxel_probability, synapses, min_ijk, max_ijk, presyn_pref):
    '''Find appropriate cell to attach the synapse to

    Args:
        cells(DataFrame w/ IJK columns): source cells (ie: pre-GIDs)
        voxel_probability(VoxelData): probabilty of IJK being picked as pre-GIDs
        synapses(DataFrame): synapses that need to be assigned, has the `WANTED_COLS` columns
        min_ijk(tuple of 3 int): Minimum coordinates in voxel space
        max_ijk(tuple of 3 int): Maximum coordinates in voxel space
        presyn_pref(callable: (cells DataFrame, synapse) -> probability):

    '''
    L.debug('assigning synapses for %s -> %s', tuple(min_ijk), tuple(max_ijk))
    pathways = []
    for voxel_pos in it.product(range(min_ijk[I], max_ijk[I]),
                                range(min_ijk[J], max_ijk[J]),
                                range(min_ijk[K], max_ijk[K])):
        start = datetime.now()

        voxel_pos = np.array(voxel_pos, np.int)
        voxel_probability.offset = voxel_pos
        local_synapses = synapses[synapses[IJK].eq(voxel_pos).all(1)].copy()

        if len(local_synapses) == 0:
            continue

        path = assign_synapse_to_cell_voxel(
            voxel_probability, cells, local_synapses, presyn_pref)
        pathways.append(path)

        #L.debug('Voxel: %s time: %s synapses: %d',
        #        voxel_pos, datetime.now() - start, len(local_synapses))
    pathways = pd.concat(pathways, ignore_index=True)
    return pathways


def pick_synapses_voxel_by_vicinity(circuit, min_xyz, max_xyz, count, segment_pref, radius=5):
    '''if the count is really low (<100?), it's faster to only do a vicinity query'''
    random_pos = (max_xyz - min_xyz) * np.random.rand(count, 3)
    segs_df = []
    for pos in random_pos:
        segs_df.append(circuit.morph.spatial_index.q_vicinity(pos, radius))

    if len(segs_df) == 0:
        return None

    segs_df = pd.concat(segs_df, ignore_index=True)
    prob_density = segment_pref(segs_df)

    try:
        prob_density = utils.normalize_probability(prob_density)
    except utils.ErrorCloseToZero:
        return None

    picked = np.random.choice(np.arange(len(segs_df)), size=count, replace=True, p=prob_density)
    return segs_df[WANTED_COLS].iloc[picked]


def _min_max_axis(min_xyz, max_xyz):
    '''return min_xyz, max_xyz, with each component being the min/max of the inputs'''
    return np.minimum(min_xyz, max_xyz), np.maximum(min_xyz, max_xyz)


def pick_synapses_voxel(circuit, min_xyz, max_xyz, count, segment_pref):
    '''Select `count` synapses from the `circuit` that lie between `min_xyz` and `max_xyz`

    Args:
        circuit: BluePy v2 Circuit object
        min_xyz(tuple of 3 floats): Minimum coordinates in world space
        max_xyz(tuple of 3 floats): Maximum coordinates in world space
        count(int): number of synapses to return
        segment_pref(callable (df -> floats)): function to assign probabilities per segment
    Returns:
        DataFrame with `WANTED_COLS`
    '''
    segs_df = circuit.morph.spatial_index.q_window(min_xyz, max_xyz)

    for k, st, en in zip('xyz', START_COLS, END_COLS):
        segs_df[k] = (segs_df[st] + segs_df[en]) / 2.

    min_xyz, max_xyz = _min_max_axis(min_xyz.copy(), max_xyz.copy())

    in_bb = ((min_xyz[0] < segs_df['x']) & (segs_df['x'] < max_xyz[0]) &
             (min_xyz[1] < segs_df['y']) & (segs_df['y'] < max_xyz[1]) &
             (min_xyz[2] < segs_df['z']) & (segs_df['z'] < max_xyz[2]))

    segs_df = segs_df[in_bb]

    if not len(segs_df[START_COLS]):
        return None

    diff = segs_df[END_COLS].values.astype(np.float) - segs_df[START_COLS].values.astype(np.float)
    segs_df['segment_length'] = np.linalg.norm(diff, axis=1)

    prob_density = segment_pref(segs_df)

    try:
        prob_density = utils.normalize_probability(prob_density)
    except utils.ErrorCloseToZero:
        L.warning('segment_pref: %s returned a prob. dist. that was too close to zero',
                  segment_pref)
        return None

    picked = np.random.choice(np.arange(len(segs_df)), size=count, replace=True, p=prob_density)

    return segs_df[WANTED_COLS].iloc[picked]


def generate_ijk_counts(min_ijk, max_ijk, voxel_synapse_count):
    '''Generator that returns the the IJK coordinates and the count given

    Args:
        min_ijk(tuple of 3 int): Minimum coordinates in voxel space
        max_ijk(tuple of 3 int): Maximum coordinates in voxel space
        voxel_synapse_count(voxcell.VoxelData): number of synapses in each voxel

    Returns:
        tuple(ijk, min_xyz, max_xyz, count)
    '''
    voxel_dimensions = voxel_synapse_count.voxel_dimensions
    offset = (voxel_synapse_count.offset // voxel_dimensions).astype(np.int)
    for ijk in it.product(range(min_ijk[I], max_ijk[I]),
                          range(min_ijk[J], max_ijk[J]),
                          range(min_ijk[K], max_ijk[K])):

        ijk = np.array(ijk)
        position = tuple(ijk - offset)
        count = voxel_synapse_count.raw[position]

        if not count:
            continue

        min_xyz = np.multiply(ijk, voxel_dimensions) - voxel_dimensions / 2
        max_xyz = min_xyz + voxel_dimensions

        yield ijk, min_xyz, max_xyz, count


def pick_synapses(circuit, voxel_synapse_count, min_ijk, max_ijk, segment_pref):
    '''For all voxels in `min_ijk` to `max_ijk` pick the synapses

    Args:
        circuit: BluePy Circuit object
        voxel_synapse_count(voxcell.VoxelData): number of synapses in each voxel
        min_ijk(tuple of 3 int): Minimum coordinates in voxel space (inclusive)
        max_ijk(tuple of 3 int): Maximum coordinates in voxel space (inclusive)
        segment_pref(callable (df -> floats)): function to assign probabilities per segment
    '''
    picked_synapses = []
    for ijk, min_xyz, max_xyz, count in generate_ijk_counts(min_ijk, max_ijk, voxel_synapse_count):
        start = datetime.now()

        syns = pick_synapses_voxel(circuit, min_xyz, max_xyz, count, segment_pref)

        if syns is None:
            continue

        syns[IJK] = pd.DataFrame([ijk], index=syns.index)
        picked_synapses.append(syns)

        #L.debug('Voxel: %s time: %s', ijk, datetime.now() - start)

    synapses = pd.concat(picked_synapses, ignore_index=True)
    return synapses
