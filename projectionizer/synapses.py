'''Functions for picking synpases'''
import json
import logging
import os
from functools import partial
from itertools import islice

import numpy as np
import pandas as pd
import voxcell
from bluepy.v2.enums import Section, Segment
from neurom import NeuriteType
from tqdm import tqdm

from projectionizer.utils import (ErrorCloseToZero, in_bounding_box,
                                  map_parallelize, mask_by_region,
                                  normalize_probability)

L = logging.getLogger(__name__)

SEGMENT_START_COLS = [Segment.X1, Segment.Y1, Segment.Z1, ]
SEGMENT_END_COLS = [Segment.X2, Segment.Y2, Segment.Z2, ]

WANTED_COLS = ['gid', Section.ID, Segment.ID, 'segment_length',
               'x', 'y', 'z', ] + SEGMENT_START_COLS + SEGMENT_END_COLS


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


def build_synapses_CA3_CA1(synapse_density, voxel_path, prefix, oversampling):
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


def pick_synapses_voxel(xyz_counts, circuit, segment_pref):
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
    min_xyz, max_xyz, count = xyz_counts

    segs_df = circuit.morph.spatial_index.q_window(min_xyz, max_xyz)

    means = 0.5 * (segs_df[SEGMENT_START_COLS].values +
                   segs_df[SEGMENT_END_COLS].values)
    segs_df = pd.concat([segs_df, pd.DataFrame(means, columns=list('xyz'))], axis=1)
    in_bb = in_bounding_box(*_min_max_axis(min_xyz, max_xyz), df=segs_df)
    segs_df = segs_df[in_bb]

    if not segs_df[SEGMENT_START_COLS].size:
        return None

    segs_df['segment_length'] = np.linalg.norm(
        (segs_df[SEGMENT_END_COLS].values.astype(np.float) -
         segs_df[SEGMENT_START_COLS].values.astype(np.float)),
        axis=1)

    prob_density = segment_pref(segs_df)
    try:
        prob_density = normalize_probability(prob_density)
    except ErrorCloseToZero:
        return None

    picked = np.random.choice(np.arange(len(segs_df)), size=count, replace=True, p=prob_density)

    return segs_df[WANTED_COLS].iloc[picked]


def pick_synapses(circuit, synapse_counts, n_islice):
    '''Sample segments from circuit
    Args:
        circuit(Circuit): The circuit to sample segment from
        synapse_counts(VoxelData):
            A VoxelData containing the number of segment to be sampled in each voxel
        n_islice(int|None):
            Number of voxels to fill, default to None=all. To be used for testing purposes.

    Returns:
        a DataFrame with the following columns:
            ['tgid', 'Section.ID', 'Segment.ID', 'segment_length', 'x', 'y', 'z']
    '''

    idx = np.nonzero(synapse_counts.raw)

    min_xyzs = synapse_counts.indices_to_positions(np.transpose(idx))
    max_xyzs = min_xyzs + synapse_counts.voxel_dimensions

    xyz_counts = list(islice(zip(min_xyzs, max_xyzs, synapse_counts.raw[idx]), n_islice))

    synapses = map_parallelize(partial(pick_synapses_voxel,
                                       circuit=circuit,
                                       segment_pref=segment_pref_length),
                               tqdm(xyz_counts))
    n_none_dfs = sum(df is None for df in synapses)
    percentage_none = n_none_dfs / float(len(synapses)) * 100
    if percentage_none > 20.:
        L.warning('%s of dataframes are None.', percentage_none)

    L.debug('Picking finished. Now concatenating...')
    return pd.concat(synapses)
