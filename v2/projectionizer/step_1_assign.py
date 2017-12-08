'''Luigi tasks that given segments, assign them to the fibers'''
import logging

import luigi
import pandas as pd

from projectionizer.step_0_sample import SampleChunkTask, VoxelSynapseCountTask
from projectionizer.utils import (CommonParams, _write_feather, load_all,
                                  )
from projectionizer.fibers import (IJK, XYZ, FIBER_COLS,
                                   closest_fibers_per_voxel,
                                   assign_synapse_fiber,
                                   )


L = logging.getLogger(__name__)


class VirtualFibersTask(CommonParams):
    '''returns a DataFrame with columns ['x', 'y', 'z', 'u', 'v', 'w']
    containing the starting position and direction of each fiber
    '''

    def run(self):
        if self.geometry in ('s1hl', 's1', ):
            from projectionizer.sscx import load_s1_virtual_fibers
            df = load_s1_virtual_fibers(self.geometry, self.voxel_path, self.prefix)
        elif self.geometry == 'hex':
            from examples.SSCX_Thalamocortical_VPM_hex import get_minicol_virtual_fibers
            df = get_minicol_virtual_fibers()
        _write_feather(self.output().path, df)

    def output(self):
        name = '{}/virtual-fibers.feather'.format(self.folder)
        return luigi.local_target.LocalTarget(name)


class ClosestFibersPerVoxel(CommonParams):
    """Return a DataFrame with the ID of the `closest_count` fibers for each voxel"""
    closest_count = luigi.IntParameter()

    def requires(self):
        return self.clone(VoxelSynapseCountTask), self.clone(VirtualFibersTask)

    def run(self):
        voxels, fibers = load_all(self.input())
        res = closest_fibers_per_voxel(voxels, fibers, self.closest_count)
        _write_feather(self.output().path, res)

    def output(self):
        name = '{}/closest-fibers-per-voxel.feather'.format(self.folder)
        return luigi.local_target.LocalTarget(name)


class SynapseIndicesTask(CommonParams):
    """Return a DataFrame with the voxels indices (i,j,k) into which each synapse is"""
    chunk_num = luigi.IntParameter()

    def requires(self):
        return self.clone(VoxelSynapseCountTask), self.clone(SampleChunkTask)

    def run(self):
        voxels, synapses = load_all(self.input())
        data = voxels.positions_to_indices(synapses[XYZ].values)
        res = pd.DataFrame(data, columns=IJK)
        _write_feather(self.output().path, res)

    def output(self):
        name = '{}/synapse_voxel_indices_{}.feather'.format(self.folder,
                                                            self.chunk_num)
        return luigi.local_target.LocalTarget(name)


class CandidateFibersPerSynapse(CommonParams):
    """Returns a DataFrame with the ID of the 25 closest fibers for each synapse"""
    chunk_num = luigi.IntParameter()

    def requires(self):
        return (self.clone(ClosestFibersPerVoxel),
                self.clone(SynapseIndicesTask),
                self.clone(SampleChunkTask),
                )

    def run(self):
        closest_fibers_per_vox, synapses_indices, synapse_position = load_all(self.input())
        synapse_position = synapse_position[XYZ]

        L.debug('Joining the synapses with their potential fibers')
        candidates = pd.merge(synapses_indices, closest_fibers_per_vox,
                              how='left', on=IJK).fillna(-1)
        # Pandas bug: merging change dtypes to float.
        # Should be solved soon:
        # http://pandas-docs.github.io/pandas-docs-travis/whatsnew.html#merging-changes
        candidates.loc[:, FIBER_COLS] = candidates.loc[:, FIBER_COLS].astype(int)
        del synapses_indices

        L.debug('Joining the synapse position')
        assert len(candidates) == len(synapse_position)

        candidates = (candidates
                      .reset_index(drop=True)
                      .join(synapse_position.reset_index(drop=True))
                      )
        candidates.drop(IJK, inplace=True, axis=1)
        _write_feather(self.output().path, candidates)

    def output(self):
        name = '{}/candidate_fibers_per_synapse_{}.feather'.format(self.folder,
                                                                   self.chunk_num)
        return luigi.local_target.LocalTarget(name)


class FiberAssignementTask(CommonParams):
    """Returns a DataFrame containing the ID of the fiber associated to each synapse"""
    chunk_num = luigi.IntParameter()
    sigma = luigi.FloatParameter()

    def requires(self):
        return self.clone(CandidateFibersPerSynapse), self.clone(VirtualFibersTask)

    def run(self):
        candidates, virtual_fibers = load_all(self.input())
        res = assign_synapse_fiber(candidates, virtual_fibers, self.sigma)
        _write_feather(self.output().path, res)

    def output(self):
        name = '{}/assigned-fibers-{}.feather'.format(self.folder, self.chunk_num)
        return luigi.local_target.LocalTarget(name)
