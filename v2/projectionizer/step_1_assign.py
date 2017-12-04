import logging
from os.path import join

import luigi
import numpy as np
import pandas as pd
import voxcell
from scipy.stats import norm

from projectionizer.step_0_sample import SampleChunkTask, VoxelSynapseCountTask
from projectionizer.utils import (CommonParams, _write_feather, choice, load,
                                  load_all, mask_by_region)

L = logging.getLogger(__name__)


class VirtualFibersTask(CommonParams):
    '''returns a DataFrame with columns ['x', 'y', 'z', 'u', 'v', 'w']
    containing the starting position and direction of each fiber
    '''

    def run(self):
        if self.geometry in ('s1hl', 's1', ):
            from projectionizer.sscx import REGION_INFO

            prefix = self.prefix or ''
            layer6_region = REGION_INFO[self.geometry]['layer6']
            mask = mask_by_region(layer6_region, self.voxel_path, prefix)
            distance_path = join(self.voxel_path, prefix + 'distance.nrrd')
            distance = voxcell.VoxelData.load_nrrd(distance_path)
            distance.raw[np.invert(mask)] = np.nan
            idx = np.transpose(np.nonzero(distance.raw == 0.0))
            fiber_pos = distance.indices_to_positions(idx)

            count = None #should be a parameter
            if count is not None:
                fiber_pos = fiber_pos[np.random.choice(np.arange(len(fiber_pos)), count)]

            orientation_path = join(self.voxel_path, prefix + 'orientation.nrrd')
            orientation = voxcell.OrientationField.load_nrrd(orientation_path)
            orientation.raw = orientation.raw.astype(np.int8)
            orientations = orientation.lookup(fiber_pos)
            y_vec = np.array([0, 1, 0])
            fiber_directions = -y_vec.dot(orientations)

            df = pd.DataFrame(np.hstack((fiber_pos, fiber_directions)), columns=list('xyzuvw'))
        elif self.geometry == 'hex':
            from examples.SSCX_Thalamocortical_VPM_hex import get_minicol_virtual_fibers
            df = get_minicol_virtual_fibers()
        _write_feather(self.output().path, df)

    def output(self):
        return luigi.local_target.LocalTarget('{}/virtual-fibers.feather'.format(self.folder))


def calc_distances(locations, virtual_fibers):
    '''find closest point from locations to fibers

    virtual_fibers is a Nx6 matrix, w/ 0:3 being the start positions,
    and 3:6 being the direction vector
    '''
    locations_count = len(locations)
    virtual_fiber_count = len(virtual_fibers)

    starts = virtual_fibers[:, 0:3]
    directions = virtual_fibers[:, 3:6]
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

    starts = np.repeat(starts, len(locations), axis=0)
    directions = np.repeat(directions, len(locations), axis=0)
    locations = np.matlib.repmat(locations, virtual_fiber_count, 1)

    distances = np.linalg.norm(np.cross((locations - starts), directions), axis=1)

    distances = distances.reshape(virtual_fiber_count, locations_count)
    return distances.T


def closest_fibers_per_voxel(synapse_counts,
                             virtual_fibers,
                             closest_count):
    '''for each occupied voxel in `synapse_counts`, find the `closest_count` number
    of virtual fibers to it

    Returns:
        dict(tuple(i, j, k) voxel -> idx into virtual_fibers
    '''
    L.debug('closest_fibers_per_voxel...')
    ijks = np.transpose(np.nonzero(synapse_counts.raw))
    pos = synapse_counts.indices_to_positions(ijks)
    pos += synapse_counts.voxel_dimensions / 2.
    distances = calc_distances(pos, virtual_fibers[list('xyzuvw')].values)

    closest_count = min(closest_count, distances.shape[1] - 1)

    # get closest_count closest virtual fibers
    partition = np.argpartition(distances, closest_count, axis=1)[:, :closest_count]
    fibers = pd.DataFrame(virtual_fibers.index[partition].values)
    return pd.concat([fibers, pd.DataFrame(ijks, columns=['i', 'j', 'k'])], axis=1)


class ClosestFibersPerVoxel(CommonParams):
    """Return a DataFrame with the ID of the 25 closest fiber for each voxel"""
    closest_count = luigi.IntParameter()

    def requires(self):
        return self.clone(VoxelSynapseCountTask), self.clone(VirtualFibersTask)

    def run(self):
        voxels, fibers = load_all(self.input())
        res = closest_fibers_per_voxel(voxels, fibers, self.closest_count)
        _write_feather(self.output().path, res)

    def output(self):
        return luigi.local_target.LocalTarget('{}/closest-fibers-per-voxel.feather'.format(self.folder))


class SynapseIndicesTask(CommonParams):
    """Return a DataFrame with the voxels indices (i,j,k) into which each synapse is"""
    chunk_num = luigi.IntParameter()

    def requires(self):
        return self.clone(VoxelSynapseCountTask), self.clone(SampleChunkTask)

    def run(self):
        voxels, synapses = load_all(self.input())
        data = voxels.positions_to_indices(synapses[list('xyz')].values)
        res = pd.DataFrame(data, columns=['i', 'j', 'k'])
        _write_feather(self.output().path, res)

    def output(self):
        return luigi.local_target.LocalTarget('{}/synapse_voxel_indices_{}.feather'.format(self.folder, self.chunk_num))


class CandidateFibersPerSynapse(CommonParams):
    """Returns a DataFrame with the ID of the 25 closest fibers for each synapse"""
    chunk_num = luigi.IntParameter()

    def requires(self):
        return self.clone(ClosestFibersPerVoxel), self.clone(SynapseIndicesTask), self.clone(SampleChunkTask)

    def run(self):
        closest_fibers_per_voxel, synapses_indices, synapse_position = load_all(self.input())
        synapse_position = synapse_position[list('xyz')]

        L.debug('Joining the synapses with their potential fibers')
        candidates = pd.merge(synapses_indices, closest_fibers_per_voxel,
                              how='left', on=['i', 'j', 'k']).fillna(-1)
        # Pandas bug: merging change dtypes to float.
        # Should be solved soon:
        # http://pandas-docs.github.io/pandas-docs-travis/whatsnew.html#merging-changes
        candidates.loc[:, map(str, range(25))] = candidates.loc[:, map(str, range(25))].astype(int)
        del synapses_indices
        L.debug('Joining the synapse position')
        assert len(candidates) == len(synapse_position)
        candidates = candidates.reset_index(drop=True).join(synapse_position.reset_index(drop=True))
        len('candidates: {}'.format(len(candidates)))
        candidates.drop(['i', 'j', 'k'], inplace=True, axis=1)
        print(len(candidates))
        _write_feather(self.output().path, candidates)

    def output(self):
        return luigi.local_target.LocalTarget('{}/candidate_fibers_per_synapse_{}.feather'.format(self.folder, self.chunk_num))


def calc_distances_vectorized(candidates, virtual_fibers):
    '''For every synapse compute the distance to each candidate fiber'''
    idx = candidates.loc[:, map(str, range(25))].fillna(0).values.astype(int)
    fiber_coord = virtual_fibers[list('xyzuvw')].values[idx]
    starts = fiber_coord[:, :, 0:3]
    directions = fiber_coord[:, :, 3:6]
    synapse_position = candidates.loc[:, ['x', 'y', 'z']].values
    distance_to_start = (synapse_position - starts.transpose(1, 0, 2)).transpose(1, 0, 2)
    return np.linalg.norm(np.cross(distance_to_start, directions), axis=2)


def assign_synapse_fiber(candidates,
                         virtual_fibers,
                         sigma):
    '''
    Assign each synapse with a closeby fiber.
    The probability of pairing follows a Normal law with the distance between
    the synapse and the fiber

    Args:
        candidates(np.arraya of Nx3): xyz positions of synapses
        virtual_fibers(np.array Nx6): point and direction vectors of virtual_fibers
        sigma(float): used for normal distribution
    '''

    distances = calc_distances_vectorized(candidates, virtual_fibers)
    # want to choose the 'best' one based on a normal distribution based on distance
    prob = norm.pdf(distances, 0, sigma)
    prob = np.nan_to_num(prob)

    idx = choice(prob)
    sgids = candidates.loc[:, map(str, range(25))].values[np.arange(len(idx)), idx]
    return pd.DataFrame({'sgid': sgids})


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
        return luigi.local_target.LocalTarget('{}/assigned-fibers-{}.feather'.format(self.folder, self.chunk_num))
