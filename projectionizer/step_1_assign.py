"""Luigi tasks that given segments, assign them to the fibers"""

import logging

import numpy as np
import pandas as pd
from bluepy import Circuit
from luigi import FloatParameter, IntParameter

from projectionizer.luigi_utils import CsvTask, FeatherTask
from projectionizer.step_0_sample import Height, SampleChunk, VoxelSynapseCount
from projectionizer.straight_fibers import (
    assign_synapse_fiber,
    candidate_fibers_per_synapse,
    closest_fibers_per_voxel,
)
from projectionizer.utils import IJK, XYZ, load, load_all, mask_by_region, write_feather

L = logging.getLogger(__name__)


class VirtualFibers(CsvTask):
    """writes a DataFrame with columns ['sgid', 'x', 'y', 'z', 'u', 'v', 'w', 'apron']
    containing the starting position and direction of each fiber

    Note: apron is a boolean indicating if the fiber is in the apron or not
    """

    def requires(self):  # pragma: no cover
        return self.clone(Height)

    def run(self):  # pragma: no cover
        height = load(self.input().path)

        def is_fiber_outside_region(df, mask):
            """Check which fibers are not in region"""
            mask_xz = mask.any(axis=1)
            idx = height.positions_to_indices(df[XYZ].to_numpy())
            return np.invert(mask_xz[tuple(idx[:, [0, 2]].T)])

        atlas = Circuit(self.circuit_config).atlas
        fibers = load(self.fiber_locations_path)
        fibers = fibers.reset_index()
        mask = mask_by_region(self.get_regions(), atlas)
        fibers["apron"] = is_fiber_outside_region(fibers, mask)

        fibers.to_csv(self.output().path, index_label="sgid")


class ClosestFibersPerVoxel(FeatherTask):
    """Return a DataFrame with the ID of the `closest_count` fibers for each voxel"""

    closest_count = IntParameter()

    def requires(self):  # pragma: no cover
        return self.clone(VoxelSynapseCount), self.clone(VirtualFibers)

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
        return (
            self.clone(ClosestFibersPerVoxel),
            self.clone(SynapseIndices),
            self.clone(SampleChunk),
        )

    def run(self):  # pragma: no cover
        closest_fibers_per_vox, synapses_indices, synapse_position = load_all(self.input())
        candidates = candidate_fibers_per_synapse(
            synapse_position[XYZ], synapses_indices, closest_fibers_per_vox
        )
        write_feather(self.output().path, candidates)


class FiberAssignment(FeatherTask):
    """Returns a DataFrame containing the ID of the fiber associated to each synapse"""

    chunk_num = IntParameter()
    sigma = FloatParameter()

    def requires(self):  # pragma: no cover
        return self.clone(CandidateFibersPerSynapse), self.clone(VirtualFibers)

    def run(self):  # pragma: no cover
        candidates, virtual_fibers = load_all(self.input())
        sgids = assign_synapse_fiber(candidates, virtual_fibers, self.sigma)
        write_feather(self.output().path, sgids)
