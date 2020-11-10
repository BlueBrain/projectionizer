'''Luigi tasks that given segments, assign them to the fibers'''
import logging

import pandas as pd
from luigi import FloatParameter, IntParameter
from bluepy.v2 import Circuit

from projectionizer.straight_fibers import (assign_synapse_fiber,
                                            closest_fibers_per_voxel,
                                            candidate_fibers_per_synapse
                                            )
from projectionizer.luigi_utils import CsvTask, FeatherTask
from projectionizer.step_0_sample import SampleChunk, VoxelSynapseCount, Height, Regions
from projectionizer.utils import load_all, write_feather, IJK, XYZ, mask_by_region

L = logging.getLogger(__name__)


class VirtualFibersNoOffset(CsvTask):
    '''writes a DataFrame with columns ['sgid', 'x', 'y', 'z', 'u', 'v', 'w', 'apron']
    containing the starting position and direction of each fiber

    Note: apron is a bool indicating if the fiber is in the apron or not
    '''
    def requires(self):  # pragma: no cover
        return self.clone(Height), self.clone(Regions)

    def run(self):  # pragma: no cover
        height, regions = load_all(self.input())
        if self.hex_fiber_locations is None:
            from projectionizer.sscx import load_s1_virtual_fibers
            atlas = Circuit(self.circuit_config).atlas
            df = load_s1_virtual_fibers(atlas, regions)
            df['apron'] = False
        else:
            from projectionizer.sscx_hex import get_minicol_virtual_fibers
            locations_path = self.load_data(self.hex_fiber_locations)
            mask = mask_by_region(regions, self.voxel_path)
            df = get_minicol_virtual_fibers(apron_bounding_box=self.hex_apron_bounding_box,
                                            height=height,
                                            region_mask=mask,
                                            locations_path=locations_path)
        df.to_csv(self.output().path, index_label='sgid')


class ClosestFibersPerVoxel(FeatherTask):
    """Return a DataFrame with the ID of the `closest_count` fibers for each voxel"""
    closest_count = IntParameter()

    def requires(self):  # pragma: no cover
        return self.clone(VoxelSynapseCount), self.clone(VirtualFibersNoOffset)

    def run(self):  # pragma: no cover
        voxels, fibers = load_all(self.input())
        res = closest_fibers_per_voxel(voxels, fibers, self.closest_count)
        write_feather(self.output().path, res)


class SynapseIndices(FeatherTask):
    """Return a DataFrame with the voxels indices (i,j,k) into which each synapse is"""
    chunk_num = IntParameter()

    def requires(self):  # pragma: no cover
        return self.clone(VoxelSynapseCount), self.clone(SampleChunk)

    def run(self):  # pragma: no cover
        voxels, synapses = load_all(self.input())
        data = voxels.positions_to_indices(synapses[XYZ].values)
        res = pd.DataFrame(data, columns=IJK)
        write_feather(self.output().path, res)


class CandidateFibersPerSynapse(FeatherTask):
    """Returns a DataFrame with the ID of the closest fibers for each synapse"""
    chunk_num = IntParameter()

    def requires(self):  # pragma: no cover
        return (self.clone(ClosestFibersPerVoxel),
                self.clone(SynapseIndices),
                self.clone(SampleChunk),)

    def run(self):  # pragma: no cover
        closest_fibers_per_vox, synapses_indices, synapse_position = load_all(
            self.input())
        candidates = candidate_fibers_per_synapse(
            synapse_position[XYZ], synapses_indices, closest_fibers_per_vox)
        write_feather(self.output().path, candidates)


class FiberAssignment(FeatherTask):
    """Returns a DataFrame containing the ID of the fiber associated to each synapse"""
    chunk_num = IntParameter()
    sigma = FloatParameter()

    def requires(self):  # pragma: no cover
        return self.clone(CandidateFibersPerSynapse), self.clone(VirtualFibersNoOffset)

    def run(self):  # pragma: no cover
        candidates, virtual_fibers = load_all(self.input())
        sgids = assign_synapse_fiber(candidates, virtual_fibers, self.sigma)
        write_feather(self.output().path, sgids)
