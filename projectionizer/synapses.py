'''Functions for picking synpases'''
import json
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import voxcell
from bluepy.v2.enums import Section, Segment
from bluepy.v2.index import SegmentIndex
from neurom import NeuriteType
from tqdm import tqdm

from projectionizer.utils import (ErrorCloseToZero, in_bounding_box,
                                  map_parallelize, mask_by_region,
                                  normalize_probability)

L = logging.getLogger(__name__)

SEGMENT_START_COLS = [Segment.X1, Segment.Y1, Segment.Z1, ]
SEGMENT_END_COLS = [Segment.X2, Segment.Y2, Segment.Z2, ]

WANTED_COLS = ['gid', Section.ID, Segment.ID, 'segment_length', 'synapse_offset',
               'x', 'y', 'z', ]


def segment_pref_length(df):
    '''don't want axons, assign probability of 0 to them, and 1 to other neurite types,
    multiplied by the length of the segment
    this will be normalized by the caller
    '''
    return df['segment_length'] * (df[Section.NEURITE_TYPE] != NeuriteType.axon).astype(float)


def build_synapses_default(height, synapse_density, oversampling):
    '''Build voxel count from densities according to the height along the column'''
    raw = np.zeros_like(height.raw, dtype=np.uint)  # pylint: disable=no-member

    voxel_volume = np.prod(np.abs(height.voxel_dimensions))
    for dist in synapse_density:
        for (bottom, density), (top, _) in zip(dist[:-1], dist[1:]):
            idx = np.nonzero((bottom <= height.raw) & (height.raw < top))
            raw[idx] = int(voxel_volume * density * oversampling)

    return height.with_data(raw)


def build_synapses_CA3_CA1(synapse_density, voxel_path, prefix, oversampling):  # pragma: no cover
    '''Build voxel count from densities according to regions'''
    atlas = voxcell.VoxelData.load_nrrd(os.path.join(voxel_path, prefix + 'brain_regions.nrrd'))
    raw = np.zeros_like(atlas.raw, dtype=np.uint)
    with open(os.path.join(voxel_path, 'hierarchy.json')) as fd:
        region_data = json.load(fd)

    for region in region_data['children']:
        for sub_region in region['children']:
            mask = mask_by_region([sub_region['id']], voxel_path, prefix)
            raw[mask] = int(synapse_density[sub_region['name']] * atlas.voxel_volume * oversampling)
    return atlas.with_data(raw)


def _min_max_axis(min_xyz, max_xyz):
    '''get min/max axis'''
    return np.minimum(min_xyz, max_xyz), np.maximum(min_xyz, max_xyz)


def _sample_with_flat_index(index_path, min_xyz, max_xyz):
    '''use flat index to get segments within min_xyz, max_xyz'''
    import libFLATIndex as FI
    try:
        index = FI.loadIndex(str(os.path.join(index_path, 'SEGMENT')))  # pylint: disable=no-member
        min_xyz_ = tuple(map(float, min_xyz))
        max_xyz_ = tuple(map(float, max_xyz))
        segs_df = FI.numpy_windowQuery(index, *(min_xyz_ + max_xyz_))  # pylint: disable=no-member
        segs_df = SegmentIndex._wrap_result(segs_df)  # pylint: disable=protected-access
        FI.unLoadIndex(index)  # pylint: disable=no-member
        del index
    except Exception:  # pylint: disable=broad-except
        return None

    return segs_df


def pick_synapses_voxel(xyz_counts, index_path, segment_pref, dataframe_cleanup):
    '''Select `count` synapses from the circuit that lie between `min_xyz` and `max_xyz`

    Args:
        xyz_counts(tuple of min_xyz, max_xyz, count): bounding box and count of synapses desired
        index_path(str): absolute path to circuit path, where a SEGMENT exists
        segment_pref(callable (df -> floats)): function to assign probabilities per segment
        dataframe_cleanup(callable (df -> df)): function to remove any unnecessary columns
        and do other processing, *must do all operations in place*, None if not needed
    Returns:
        DataFrame with `WANTED_COLS`
    '''
    min_xyz, max_xyz, count = xyz_counts

    segs_df = _sample_with_flat_index(index_path, min_xyz, max_xyz)

    if segs_df is None:
        return None

    starts = segs_df[SEGMENT_START_COLS].values
    ends = segs_df[SEGMENT_END_COLS].values

    # pick point somewhere along the segment
    alpha = np.random.random_sample((len(segs_df), 1))
    locations = alpha * starts + (1. - alpha) * ends
    locations = pd.DataFrame(locations, columns=list('xyz'))
    in_bb = in_bounding_box(*_min_max_axis(min_xyz, max_xyz), df=locations)
    segs_df = segs_df.join(locations)[in_bb].copy()

    if not segs_df[SEGMENT_START_COLS].size:
        return None

    starts = segs_df[SEGMENT_START_COLS].values.astype(np.float)
    ends = segs_df[SEGMENT_END_COLS].values.astype(np.float)
    segs_df['segment_length'] = np.linalg.norm(ends - starts, axis=1)
    segs_df['synapse_offset'] = alpha[in_bb].ravel() * segs_df['segment_length']

    prob_density = segment_pref(segs_df)
    try:
        prob_density = normalize_probability(prob_density)
    except ErrorCloseToZero:
        return None

    picked = np.random.choice(np.arange(len(segs_df)), size=count, replace=True, p=prob_density)

    segs_df = segs_df[WANTED_COLS].iloc[picked]

    if dataframe_cleanup is not None:
        dataframe_cleanup(segs_df)

    return segs_df


def pick_synapses(index_path, synapse_counts,
                  segment_pref=segment_pref_length, dataframe_cleanup=None):
    '''Sample segments from circuit
    Args:
        index_path: absolute path to circuit path, where a SEGMENT exists
        synapse_counts(VoxelData):
            A VoxelData containing the number of segment to be sampled in each voxel

    Returns:
        a DataFrame with the following columns:
            ['tgid', 'Section.ID', 'Segment.ID', 'segment_length', 'x', 'y', 'z']
    '''

    idx = np.nonzero(synapse_counts.raw)

    min_xyzs = synapse_counts.indices_to_positions(np.transpose(idx))
    max_xyzs = min_xyzs + synapse_counts.voxel_dimensions

    xyz_counts = zip(min_xyzs, max_xyzs, synapse_counts.raw[idx])

    func = partial(pick_synapses_voxel,
                   index_path=index_path,
                   segment_pref=segment_pref,
                   dataframe_cleanup=dataframe_cleanup)

    synapses = list(map_parallelize(func, tqdm(xyz_counts)))

    n_none_dfs = sum(df is None for df in synapses)
    percentage_none = n_none_dfs / float(len(synapses)) * 100
    if percentage_none > 20.:  # pragma: no cover
        L.warning('%s of dataframes are None.', percentage_none)

    L.debug('Picking finished. Now concatenating...')
    return pd.concat(synapses, ignore_index=True)


def organize_indices(synapses):
    '''*inplace* reorganize the synapses indices'''
    synapses.set_index(['tgid', 'sgid'], inplace=True)
    synapses.sort_index(inplace=True)
    synapses.reset_index(inplace=True)

    return synapses
