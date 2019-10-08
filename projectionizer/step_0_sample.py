'''Step 0; sample segments from circuit to be used as potential synapses
'''
import json
import os

import numpy as np
import pandas as pd
import voxcell

from luigi import BoolParameter, FloatParameter, IntParameter, ListParameter
from projectionizer.luigi_utils import FeatherTask, JsonTask, NrrdTask
from projectionizer.sscx import REGION_INFO, recipe_to_height_and_density
from projectionizer.synapses import (build_synapses_default,
                                     pick_synapses)
from projectionizer.utils import load, load_all, mask_by_region, write_feather


class VoxelSynapseCount(NrrdTask):  # pragma: no cover
    """Generate the VoxelData containing the number
    of segment to be sampled in each voxel"""
    oversampling = FloatParameter()

    def requires(self):
        return self.clone(Height), self.clone(SynapseDensity)

    def run(self):
        height, synapse_density = load_all(self.input())
        res = build_synapses_default(
            height, synapse_density, self.oversampling)
        res.save_nrrd(self.output().path)


class Height(NrrdTask):  # pragma: no cover
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
            from projectionizer.sscx_hex import voxel_space
            max_height = sum(h for _, h in self.layers)  # pylint: disable=not-an-iterable
            fiber_locations = self.load_data(self.hex_fiber_locations)
            voxels = voxel_space(hex_edge_len=self.hex_side,
                                 locations_path=fiber_locations,
                                 max_height=max_height
                                 )
            xyz = voxels.indices_to_positions(np.indices(
                voxels.raw.shape).transpose(1, 2, 3, 0) + (0.5, 0.5, 0.5))
            distance = voxels.with_data(xyz[:, :, :, 1])
        else:
            raise Exception('Unknown geometry: {}'.format(self.geometry))
        distance.save_nrrd(self.output().path)


class SampleChunk(FeatherTask):
    """Split the big sample into chunks"""
    chunk_num = IntParameter()

    def requires(self):
        return self.clone(FullSample)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        full_sample = load(self.input().path)
        chunk_size = int((len(full_sample) / self.n_total_chunks) + 1)
        start, end = self.chunk_num * chunk_size, (self.chunk_num + 1) * chunk_size
        chunk_df = full_sample.iloc[start:end]
        write_feather(self.output().path, chunk_df)


class FullSample(FeatherTask):  # pragma: no cover
    '''Sample segments from circuit
    '''
    n_slices = IntParameter()
    from_chunks = BoolParameter(default=False)

    def requires(self):
        if self.from_chunks:
            return [self.clone(SampleChunk, chunk_num=i) for i in range(self.n_total_chunks)]
        return self.clone(VoxelSynapseCount)

    def run(self):
        if self.from_chunks:
            # pylint: disable=maybe-no-member
            chunks = load(self.input().path)
            write_feather(self.output().path, pd.concat(chunks))
        else:
            # pylint: disable=maybe-no-member
            voxels = load(self.input().path)
            circuit_path = os.path.dirname(self.circuit_config)
            synapses = pick_synapses(circuit_path, voxels)

            synapses.rename(columns={'gid': 'tgid'}, inplace=True)
            write_feather(self.output().path, synapses)


class SynapseDensity(JsonTask):  # pragma: no cover
    '''Return the synaptic density profile'''
    density_params = ListParameter()

    def run(self):
        res = [recipe_to_height_and_density(self.layers,
                                            data['low_layer'],
                                            data['low_fraction'],
                                            data['high_layer'],
                                            data['high_fraction'],
                                            data['density_profile'])
               for data in self.density_params]  # pylint: disable=not-an-iterable
        with self.output().open('w') as outfile:
            json.dump(res, outfile)
