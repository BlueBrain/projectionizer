'''Luigi tasks that given segments, assign them to the fibers'''
import logging

import numpy as np
import pandas as pd
from voxcell import VoxelData

from luigi import FloatParameter, IntParameter
from projectionizer.straight_fibers import (assign_synapse_fiber,
                                            closest_fibers_per_voxel)
from projectionizer.luigi_utils import CsvTask, FeatherTask, NrrdTask
from projectionizer.step_0_sample import SampleChunk, VoxelSynapseCount
from projectionizer.utils import choice, load_all, write_feather, IJK, XYZ

L = logging.getLogger(__name__)


LONGITUDINAL_AXIS = 0
TRANSVERSE_AXIS = 1
RADIAL_AXIS = 2


class VirtualFibersNoOffset(CsvTask):
    '''writes a DataFrame with columns ['sgid', 'x', 'y', 'z', 'u', 'v', 'w', 'apron']
    containing the starting position and direction of each fiber

    Note: apron is a bool indicating if the fiber is in the apron or not
    '''

    def run(self):  # pragma: no cover
        if self.geometry in ('s1hl', 's1', ):
            from projectionizer.sscx import load_s1_virtual_fibers
            df = load_s1_virtual_fibers(
                self.geometry, self.voxel_path, self.prefix)
            df['apron'] = False
        elif self.geometry == 'hex':
            from projectionizer.sscx_hex import get_minicol_virtual_fibers
            locations_path = self.load_data(self.hex_fiber_locations)
            df = get_minicol_virtual_fibers(apron_size=self.hex_apron_size,
                                            hex_edge_len=self.hex_side,
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
    """Returns a DataFrame with the ID of the 25 closest fibers for each synapse"""
    chunk_num = IntParameter()

    def requires(self):  # pragma: no cover
        return (self.clone(ClosestFibersPerVoxel),
                self.clone(SynapseIndices),
                self.clone(SampleChunk),)

    def run(self):  # pragma: no cover
        closest_fibers_per_vox, synapses_indices, synapse_position = load_all(
            self.input())
        synapse_position = synapse_position[XYZ]

        L.debug('Joining the synapses with their potential fibers')
        candidates = pd.merge(synapses_indices, closest_fibers_per_vox,
                              how='left', on=IJK).fillna(-1)
        # Pandas bug: merging change dtypes to float.
        # Should be solved soon:
        # http://pandas-docs.github.io/pandas-docs-travis/whatsnew.html#merging-changes
        cols = candidates.columns[:-3]
        candidates.loc[:, cols] = candidates.loc[:, cols].astype(int)
        del synapses_indices

        L.debug('Joining the synapse position')
        assert len(candidates) == len(synapse_position)

        candidates = (candidates
                      .reset_index(drop=True)
                      .join(synapse_position.reset_index(drop=True)))
        candidates.drop(IJK, inplace=True, axis=1)
        write_feather(self.output().path, candidates)


def to_cylindrical_coordinates(xyz_coordinates):
    '''Returns the coordinates in the cylindrical coordinate system (r, theta, Z) '''
    return xyz_coordinates


def distance_to_centroid(candidates, fiber_centroids):
    '''For every synapse returns the distance vector to each candidate fiber'''
    idx = candidates.loc[:, map(str, range(25))].fillna(0).values.astype(int)
    fiber_coord = fiber_centroids[['radial_mean',
                                   'transverse_mean', 'longitudinal_mean']].values[idx]
    synapse_position = to_cylindrical_coordinates(
        candidates.loc[:, ['x', 'y', 'z']].values)

    return (synapse_position - fiber_coord.transpose(1, 0, 2)).transpose(1, 0, 2)


class Centroids(FeatherTask):
    '''Build a DataFrame with the mean and std values of the 2D(longitudinal, transveral) gaussians
    that describes the probability for each fiber to make a connection
    Here is the text that describes the connection
    (from
     https://bbpteam.epfl.ch/project/issues/secure/attachment/17414/20170703_Hippocampus.pdf):

    The connection probability between CA3 PC and CA1 neuron can be described by
    three distributions, one for each hippocampal axis. The hippocampal axes are: radial axis,
    transverse axis, and longitudinal axis.

    For simplicity, a CA3 PC makes connections preferentially at the same level of
    longitudinal axis, with a probability that decreases at each tip of the axis following
    a gaussian distribution.
    The CA3 PC also has a gaussian profile along the transverse axis. The peak of the
    gaussian can occur at any point of the transverse axis with a uniform probability.
    The profile along the radial axis depends on the layer as following:
    SLM: 0.0025740628
    SR: 0.6792951748
    SP: 0.0705293209
    SO: 0.2476014415

    Note: The sus - mentionned radial profile is the one aleady used for sampling the synapses.
    For this reason we dont need to take it into account her.
    '''

    # def requires(self): # pragma: no cover
    #     return self.clone(VirtualFibersNoOffset)

    def run(self):  # pragma: no cover
        # fibers = load(self.input().path)

        size = 100
        fibers = pd.DataFrame({LONGITUDINAL_AXIS: np.random.random(size),
                               TRANSVERSE_AXIS: np.random.random(size)})

        transverse_start = 0
        transverse_end = 180
        transverse_peak_means = transverse_start + \
            np.random.random((len(fibers),)) * \
            (transverse_end - transverse_start)
        longitudinal_peak_means = fibers.loc[:, LONGITUDINAL_AXIS]
        res = pd.DataFrame({'transversal_mean': transverse_peak_means,
                            'radial_mean': 0,
                            'longitudinal_mean': longitudinal_peak_means,
                            'transversal_sigma': 0.2,
                            'radial_sigma': 1,
                            'longitudinal_sigma': 0.1, })
        write_feather(self.output().path, res)


class SynapticDistributionPerAxon(NrrdTask):
    '''4D array with the 3D synaptic density distribution of each neuron'''

    def requires(self):  # pragma: no cover
        return self.clone(Centroids), self.clone(VoxelSynapseCount)

    def run(self):  # pragma: no cover  # pylint: disable=too-many-locals
        centroids, voxels = load_all(self.input())
        xyz = voxels.indices_to_positions(np.indices(
            voxels.raw.shape).transpose(1, 2, 3, 0))
        dimension_x, dimension_y, _ = voxels.voxel_dimensions
        x = xyz[:, :, :, 0]
        y = xyz[:, :, :, 1]
        probs = list()

        for centroid in centroids:
            mean_x = centroid['longitudinal_mean']
            sigma_x = centroid['longitudinal_sigma']
            mean_y = centroid['transveral_mean']
            sigma_y = centroid['transveral_mean']
            # pylint: disable=undefined-variable
            prob = (norm.cdf(x + dimension_x, loc=mean_x, scale=sigma_x) -
                    norm.cdf(x, loc=mean_x, scale=sigma_x)) * \
                   (norm.cdf(y + dimension_y, loc=mean_y, scale=sigma_y) -
                    norm.cdf(y, loc=mean_y, scale=sigma_y))
            prob /= prob.sum()
            probs.append(prob)
        prob_array = np.stack(probs).transpose((1, 2, 3, 0))
        prob_array /= prob_array.sum(axis=3)[:, :, :, np.newaxis]
        res = VoxelData(prob_array, voxel_dimensions=(1, 1, 1, 1))
        res.save_nrrd(self.output().path)


class FiberAssignment(FeatherTask):
    """Returns a DataFrame containing the ID of the fiber associated to each synapse"""
    chunk_num = IntParameter()
    sigma = FloatParameter()

    def requires(self):  # pragma: no cover
        if self.geometry != 'CA3_CA1':
            return self.clone(CandidateFibersPerSynapse), self.clone(VirtualFibersNoOffset)
        return self.clone(SynapticDistributionPerAxon), self.clone(SynapseIndices)

    def run(self):  # pragma: no cover
        if self.geometry != 'CA3_CA1':
            candidates, virtual_fibers = load_all(self.input())
            sgids = assign_synapse_fiber(
                candidates, virtual_fibers, self.sigma)
        else:
            proba, indices = load_all(self.input())
            idx = choice(proba[indices.i, indices.j, indices.k])
            sgids = pd.DataFrame({'sgids': idx})

        write_feather(self.output().path, sgids)
