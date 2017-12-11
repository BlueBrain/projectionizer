'''Step 0; sample segments from circuit to be used as potential synapses
'''
import json
import logging
import os
from functools import partial
from itertools import islice

import luigi
import numpy as np
import pandas as pd
import voxcell
import yaml
from bluepy.v2.circuit import Circuit
from bluepy.v2.enums import Section, Segment
from luigi import FloatParameter, IntParameter, Parameter
from neurom import NeuriteType
from tqdm import tqdm

from projectionizer.sscx import REGION_INFO, recipe_to_height_and_density
from projectionizer.utils import (CommonParams, ErrorCloseToZero,
                                  _write_feather, in_bounding_box, load,
                                  load_all, map_parallelize, mask_by_region,
                                  normalize_probability)

L = logging.getLogger(__name__)

# bluepy.v2 returns a DataFrame with the start and endpoint of the segments when performing a query,
# simplify addressing them using the following

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


class VoxelSynapseCountTask(CommonParams):
    """Generate the VoxelData containing the number
    of segment to be sampled in each voxel"""
    oversampling = FloatParameter()

    def requires(self):
        return self.clone(HeightTask), self.clone(SynapseDensity)

    def run(self):
        height, synapse_density = load_all(self.input())
        raw = np.zeros_like(height.raw, dtype=np.uint)  # pylint: disable=no-member

        voxel_volume = np.prod(np.abs(height.voxel_dimensions))
        for dist in synapse_density:
            for (bottom, density), (top, _) in zip(dist[:-1], dist[1:]):
                idx = np.nonzero((bottom < height.raw) & (height.raw < top))
                raw[idx] = int(voxel_volume * density * self.oversampling)

        height.with_data(raw).save_nrrd(self.output().path)

    def output(self):
        return luigi.local_target.LocalTarget('{}/voxel-counts.nrrd'.format(self.folder))


class HeightTask(CommonParams):
    '''return a VoxelData instance w/ all the heights for given region_name

    distance is defined as from the voxel to the bottom of L6, voxels
    outside of region_name are set to 0

    Args:
        region_name(str): name to look up in atlas
        path(str): path to where nrrd files are, must include 'brain_regions.nrrd'
        prefix(str): Prefix (ie: uuid) used to identify atlas/voxel set
    '''

    def run(self):
        if self.geometry in ('s1hl', 's1', ):
            prefix = self.prefix or ''
            region = REGION_INFO[self.geometry]['region']
            mask = mask_by_region(region, self.voxel_path, prefix)
            distance = voxcell.VoxelData.load_nrrd(
                os.path.join(self.voxel_path, prefix + 'distance.nrrd'))
            distance.raw[np.invert(mask)] = 0.
        elif self.geometry == 'hex':
            from examples.SSCX_Thalamocortical_VPM_hex import voxel_space
            voxels = voxel_space()
            xyz = voxels.indices_to_positions(np.indices(
                voxels.raw.shape).transpose(1, 2, 3, 0) + (0.5, 0.5, 0.5))
            distance = voxels.with_data(xyz[:, :, :, 1])
        else:
            raise Exception('Unknown geometry: {}'.format(self.geometry))
        distance.save_nrrd(self.output().path)

    def output(self):
        name = '{}/get-heights.nrrd'.format(self.folder)
        return luigi.local_target.LocalTarget(name)


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

    def _min_max_axis(min_xyz, max_xyz):
        '''get min/max axis'''
        return np.minimum(min_xyz, max_xyz), np.maximum(min_xyz, max_xyz)

    in_bb = in_bounding_box(*_min_max_axis(min_xyz, max_xyz), df=segs_df)
    segs_df = segs_df[in_bb]

    if not len(segs_df[SEGMENT_START_COLS]):
        return None

    segs_df['segment_length'] = np.linalg.norm(
        (segs_df[SEGMENT_END_COLS].values.astype(np.float) -
         segs_df[SEGMENT_START_COLS].values.astype(np.float)),
        axis=1)

    prob_density = segment_pref(segs_df)
    try:
        prob_density = normalize_probability(prob_density)
    except ErrorCloseToZero:
        L.warning('segment_pref: %s returned a prob. dist. that was too close to zero',
                  segment_pref)
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
    L.debug(len(xyz_counts))

    synapses = map_parallelize(partial(pick_synapses_voxel,
                                       circuit=circuit,
                                       segment_pref=segment_pref_length),
                               tqdm(xyz_counts))
    L.debug('Picking finished. Now concatenating...')
    return pd.concat(synapses)


class FullSampleTask(CommonParams):
    '''Sample segments from circuit
    '''
    n_slices = IntParameter()

    def requires(self):
        return self.clone(VoxelSynapseCountTask)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        voxels = load(self.input().path)
        # Hack, cause I don't know how to pass a None IntParameter to luigi -__-
        n_slices = self.n_slices if self.n_slices > 0 else None
        synapses = pick_synapses(Circuit(self.circuit_config), voxels, n_slices)

        remove_cols = SEGMENT_START_COLS + SEGMENT_END_COLS
        synapses.drop(remove_cols, axis=1, inplace=True)
        synapses.rename(columns={'gid': 'tgid'}, inplace=True)
        _write_feather(self.output().path, synapses)

    def output(self):
        name = '{}/full_sample.feather'.format(self.folder)
        return luigi.local_target.LocalTarget(name)


class SampleChunkTask(CommonParams):
    """Split the big sample into chunks"""
    chunk_num = luigi.IntParameter()

    def requires(self):
        return self.clone(FullSampleTask)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        full_sample = load(self.input().path)
        chunk_size = (len(full_sample) / self.n_total_chunks) + 1
        start, end = np.array([self.chunk_num, self.chunk_num + 1]) * chunk_size
        chunk_df = full_sample.iloc[start: end]
        _write_feather(self.output().path, chunk_df)

    def output(self):
        name = '{}/sampling_{}.feather'.format(self.folder, self.chunk_num)
        return luigi.local_target.LocalTarget(name)


class SynapseDensity(CommonParams):
    '''Return the synaptic density profile'''
    density_params = Parameter()

    def run(self):
        density_params = yaml.load(self.density_params)
        res = [recipe_to_height_and_density(data['low_layer'], data['low_fraction'],
                                            data['high_layer'], data['high_fraction'],
                                            data['density_profile'])
               for data in density_params]
        with self.output().open('w') as outfile:
            json.dump(res, outfile)

    def output(self):
        name = '{}/synaptic_density_profile.json'.format(self.folder)
        return luigi.local_target.LocalTarget(name)
