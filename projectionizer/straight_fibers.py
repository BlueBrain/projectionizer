'''
Functions related to calculating distances to straight fibers, like the ones used in the SSCX
hex and S1
'''
from functools import partial
import logging

import numpy as np
from numpy import matlib
import pandas as pd
from scipy.stats import norm  # pylint: disable=no-name-in-module

from projectionizer.utils import choice, map_parallelize, XYZUVW, IJK, XYZ


L = logging.getLogger(__name__)

VF_STARTS = slice(0, 3)
VF_DIRS = slice(3, 6)
NOT_CANDIDATE_STARTS = -3  # Columns not storing candidates ids


def calc_distances(locations, virtual_fibers):
    '''find closest point from locations to fibers, calculate the distance to this point

    virtual_fibers is a Nx6 matrix, w/ 0:3 being the start positions,
    and 3:6 being the direction vector
    '''
    locations_count = len(locations)
    virtual_fiber_count = len(virtual_fibers)

    starts = virtual_fibers[:, VF_STARTS]
    directions = virtual_fibers[:, VF_DIRS]
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

    starts = np.repeat(starts, len(locations), axis=0)
    directions = np.repeat(directions, len(locations), axis=0)
    locations = matlib.repmat(locations, virtual_fiber_count, 1)

    distances = np.linalg.norm(np.cross((locations - starts), directions), axis=1)

    distances = distances.reshape(virtual_fiber_count, locations_count)
    return distances.T


def calc_pathlength_to_fiber_start(locations, sgid_fibers):
    '''find distance to the closest point on the `sgid_fibers`, and the distance to the `start`

    sgid_fibers is a Nx6 matrix, w/ 0:3 being the start positions, and 3:6 being the dir vector
    the direction vector is expected to be normalized
    '''
    starts = sgid_fibers[:, VF_STARTS]
    directions = sgid_fibers[:, VF_DIRS]
    hypotenuse = locations - starts
    distances = (np.linalg.norm(np.cross(hypotenuse, directions, axis=1), axis=1) +
                 np.sum(hypotenuse * directions, axis=1))

    return distances


def _closest_fibers_per_voxel(pos, virtual_fibers, closest_count):
    '''get closest_count closest virtual fibers for positions in pos'''
    distances = calc_distances(pos, virtual_fibers[XYZUVW].values)
    closest_count = min(closest_count, distances.shape[1] - 1)
    fiber_idx = np.argpartition(distances, closest_count, axis=1)[:, :closest_count]
    fibers = pd.DataFrame(virtual_fibers.index[fiber_idx].values)
    return fibers


def closest_fibers_per_voxel(synapse_counts, virtual_fibers, closest_count):
    '''for each occupied voxel in `synapse_counts`, find the `closest_count` number
    of virtual fibers to it

    Returns:
        dict(tuple(i, j, k) voxel -> idx into virtual_fibers
    '''
    L.debug('closest_fibers_per_voxel...')
    ijks = np.transpose(np.nonzero(synapse_counts.raw))
    pos = synapse_counts.indices_to_positions(ijks)

    split_count = len(pos) // 1000 + 1
    fibers = map_parallelize(partial(_closest_fibers_per_voxel,
                                     virtual_fibers=virtual_fibers,
                                     closest_count=closest_count),
                             np.array_split(pos, split_count))
    fibers = pd.concat(fibers, sort=False, ignore_index=True)

    return pd.concat([fibers, pd.DataFrame(ijks, columns=IJK)], axis=1)


def calc_distances_vectorized(candidates, virtual_fibers):
    '''For every synapse compute the distance to each candidate fiber'''
    cols = candidates.columns.difference(XYZ)
    idx = candidates.loc[:, cols].fillna(0).values.astype(int)
    fiber_coord = virtual_fibers[XYZUVW].values[idx]
    starts = fiber_coord[:, :, VF_STARTS]
    directions = fiber_coord[:, :, VF_DIRS]
    synapse_position = candidates.loc[:, XYZ].values
    distance_to_start = (synapse_position - starts.transpose(1, 0, 2)).transpose(1, 0, 2)
    return np.linalg.norm(np.cross(distance_to_start, directions), axis=2)


def candidate_fibers_per_synapse(synapse_position_xyz, synapses_indices, closest_fibers_per_vox):
    '''based on synapse location, find candidate fibers

    Args:
        synapse_position_xyz(dataframe): x/y/z positions of synapses
        synapses_indices(dataframe): i/j/k voxel positions of fibers
        closest_fibers_per_vox(dataframe): fibers close to voxels
    '''
    L.debug('Joining the synapses with their potential fibers')
    candidates = pd.merge(synapses_indices, closest_fibers_per_vox,
                          how='left', on=IJK).fillna(-1)
    # Pandas bug: merging change dtypes to float.
    # Should be solved soon:
    # http://pandas-docs.github.io/pandas-docs-travis/whatsnew.html#merging-changes
    cols = candidates.columns.difference(IJK)
    candidates.loc[:, cols] = candidates.loc[:, cols].astype(int)
    del synapses_indices

    L.debug('Joining the synapse position')
    assert len(candidates) == len(synapse_position_xyz)

    candidates = (candidates
                  .reset_index(drop=True)
                  .join(synapse_position_xyz.reset_index(drop=True)))
    candidates.drop(IJK, inplace=True, axis=1)

    return candidates


def assign_synapse_fiber(candidates, virtual_fibers, sigma,
                         distance_calculator=calc_distances_vectorized):
    '''
    Assign each synapse with a close by fiber.
    The probability of pairing follows a Normal law with the distance between
    the synapse and the fiber

    Args:
        candidates(np.arraya of Nx3): xyz positions of synapses
        virtual_fibers(np.array Nx6): point and direction vectors of virtual_fibers
        sigma(float): used for normal distribution
    '''

    distances = distance_calculator(candidates, virtual_fibers)
    # want to choose the 'best' one based on a normal distribution based on distance
    prob = norm.pdf(distances, 0, sigma)
    prob = np.nan_to_num(prob)

    idx = choice(prob)
    cols = candidates.columns.difference(XYZ)
    sgids = candidates.loc[:, cols].values[np.arange(len(idx)), idx]
    return pd.DataFrame({'sgid': sgids})
