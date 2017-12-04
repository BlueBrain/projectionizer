'''Luigi tasks that given segments, assign them to the fibers'''
import logging

import luigi
import pandas as pd
import voxcell
from luigi.local_target import LocalTarget
from scipy.stats import multivariate_normal, norm

from projectionizer.fibers import (FIBER_COLS, IJK, XYZ, assign_synapse_fiber,
                                   closest_fibers_per_voxel)
from projectionizer.step_0_sample import SampleChunkTask, VoxelSynapseCountTask
from projectionizer.utils import CommonParams, _write_feather, load, load_all

L = logging.getLogger(__name__)


LONGITUDINAL_AXIS = 0
TRANSVERSE_AXIS = 1
RADIAL_AXIS = 2


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
            df = get_minicol_virtual_fibers(self.apron)
        _write_feather(self.output().path, df)

    def output(self):
        return LocalTarget('{}/virtual-fibers.feather'.format(self.folder))


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
    """Return a DataFrame with the ID of the `closest_count` fibers for each voxel"""
    closest_count = luigi.IntParameter()

    def requires(self):
        return self.clone(VoxelSynapseCountTask), self.clone(VirtualFibersTask)

    def run(self):
        voxels, fibers = load_all(self.input())
        res = closest_fibers_per_voxel(voxels, fibers, self.closest_count)
        _write_feather(self.output().path, res)

    def output(self):
        return LocalTarget('{}/closest-fibers-per-voxel.feather'.format(self.folder))


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


def calc_distances_vectorized(candidates, virtual_fibers):
    '''For every synapse compute the distance to each candidate fiber'''
    idx = candidates.loc[:, map(str, range(25))].fillna(0).values.astype(int)
    fiber_coord = virtual_fibers[list('xyzuvw')].values[idx]
    starts = fiber_coord[:, :, 0:3]
    directions = fiber_coord[:, :, 3:6]
    synapse_position = candidates.loc[:, ['x', 'y', 'z']].values
    distance_to_start = (synapse_position - starts.transpose(1, 0, 2)).transpose(1, 0, 2)
    return np.linalg.norm(np.cross(distance_to_start, directions), axis=2)


def to_cylindrical_coordinates(xyz_coordinates):
    '''Returns the coordinates in the cylindrical coordinate system (r, theta, Z) '''
    return xyz_coordinates


def distance_to_centroid(candidates, fiber_centroids):
    '''For every synapse returns the distance vector to each candidate fiber'''
    idx = candidates.loc[:, map(str, range(25))].fillna(0).values.astype(int)
    fiber_coord = fiber_centroids[['radial_mean',
                                   'transverse_mean', 'longitudinal_mean']].values[idx]
    synapse_position = to_cylindrical_coordinates(candidates.loc[:, ['x', 'y', 'z']].values)

    return (synapse_position - fiber_coord.transpose(1, 0, 2)).transpose(1, 0, 2)


class Centroids(CommonParams):
    '''Build a DataFrame with the mean and std values of the 2D (longitudinal, transveral) gaussians that describes the probability for each fiber to make a connection
    Here is the text that describes the connection (from https://bbpteam.epfl.ch/project/issues/secure/attachment/17414/20170703_Hippocampus.pdf):

    The connection probability between CA3 PC and CA1 neuron can be described by three distributions,
    one for each hippocampal axis. The hippocampal axes are: radial axis, transverse axis, and longitudinal axis.

    For simplicity, a CA3 PC makes connections preferentially at the same level of longitudinal axis, with
    a probability that decreases at each tip of the axis following a gaussian distribution.
    The CA3 PC also has a gaussian profile along the transverse axis. The peak of the gaussian can occur at
    any point of the transverse axis with a uniform probability.
    The profile along the radial axis depends on the layer as following:
    SLM: 0.0025740628
    SR: 0.6792951748
    SP: 0.0705293209
    SO: 0.2476014415

    Note: The sus-mentionned radial profile is the one aleady used for sampling the synapses.
    For this reason we don't need to take it into account her.
    '''

    # def requires(self):
    #     return self.clone(VirtualFibersTask)

    def run(self):
        # fibers = load(self.input().path)

        size = 100
        fibers = pd.DataFrame({LONGITUDINAL_AXIS: np.random.random(size),
                               TRANSVERSE_AXIS: np.random.random(size)})

        transverse_start = 0
        transverse_end = 180
        transverse_peak_means = transverse_start + \
            np.random.random((len(fibers),)) * (transverse_end - transverse_start)
        longitudinal_peak_means = fibers.loc[:, LONGITUDINAL_AXIS]
        res = pd.DataFrame({'transverse_mean': transverse_peak_means,
                            'radial_mean': 0,
                            'longitudinal_mean': longitudinal_peak_means,
                            'transverse_sigma': 0.2,
                            'radial_sigma': 1,
                            'longitudinal_sigma': 0.1, })
        _write_feather(self.output().path, res)

    def output(self):
        return LocalTarget('{}/fiber-connection-distributions.feather'.format(self.folder))


class SynapticDistributionPerAxon(CommonParams):
    def requires(self):
        return self.clone(VoxelSynapseCountTask)

    def run(self):
        voxels = load(self.input().path)
        xyz = voxels.indices_to_positions(np.indices(
            voxels.raw.shape).transpose(1, 2, 3, 0))
        dimension_x, dimension_y, dimension_z = voxels.voxel_dimensions
        x = xyz[:, :, :, 0]
        y = xyz[:, :, :, 1]
        z = xyz[:, :, :, 2]
        probs = np.array(voxels.shape + (len(neurons),))
        for neuron in neurons:
            prob = (norm.cdf(x + dimension_x) - norm.cdf(x + dimension_x)) * \
                   (norm.cdf(y + dimension_y) - norm.cdf(y + dimension_y))
            prob /= prob.sum()


class FiberAssignementTask(CommonParams):
    """Returns a DataFrame containing the ID of the fiber associated to each synapse"""
    chunk_num = luigi.IntParameter()
    # transverse_sigma = luigi.FloatParameter()
    # longitudinal_sigma = luigi.FloatParameter()
    sigma = luigi.FloatParameter()

    def requires(self):
        # return self.clone(CandidateFibersPerSynapse), self.clone(VirtualFibersTask)
        return self.clone(Centroids)

    def run(self):
        '''
        Assign each synapse with a closeby fiber.
        The probability of pairing follows a Normal law with the distance between
        the synapse and the fiber
        '''
        # candidates, virtual_fibers = load_all(self.input())
        centroids = load(self.input().path)
        candidates = pd.DataFrame(np.random.randint(100, size=(1234, 25)),
                                  columns=[str(i) for i in range(25)])
        candidates['x'] = np.random.random()
        candidates['y'] = np.random.random()
        candidates['z'] = np.random.random()

        distances = distance_to_centroid(candidates, centroids)
        theta_Z_distances = distances[:, :, [TRANSVERSE_AXIS, LONGITUDINAL_AXIS]]
        prob = multivariate_normal.pdf(theta_Z_distances, mean=(
            0, 0), cov=np.diag([self.transverse_sigma, self.longitudinal_sigma]))

        prob = np.nan_to_num(prob)

        idx = choice(prob)
        sgids = pd.DataFrame(candidates.loc[:, map(str, range(25))].values[np.arange(len(idx)), idx],
                             columns=['sgid'])

        _write_feather(self.output().path, sgids)

    def output(self):
        name = '{}/assigned-fibers-{}.feather'.format(self.folder, self.chunk_num)
        return luigi.local_target.LocalTarget(name)
