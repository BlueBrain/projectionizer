'''
Functions related to calculating distances to straight fibers, like the ones used in the SSCX
hex and S1
'''
import logging

import numpy as np
import pandas as pd

from scipy.stats import norm  # pylint: disable=no-name-in-module

from projectionizer.utils import choice

L = logging.getLogger(__name__)

XYZUVW = list('xyzuvw')
IJK = list('ijk')
XYZ = list('xyz')
FIBER_COLS = map(str, range(25))  # TODO: this shouldn't be hard-coded

VF_STARTS = slice(0, 3)
VF_DIRS = slice(3, 6)


def calc_distances(locations, virtual_fibers):
    '''find closest point from locations to fibers

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
    locations = np.matlib.repmat(locations, virtual_fiber_count, 1)

    distances = np.linalg.norm(np.cross((locations - starts), directions), axis=1)

    distances = distances.reshape(virtual_fiber_count, locations_count)
    return distances.T


def closest_fibers_per_voxel(synapse_counts, virtual_fibers, closest_count):
    '''for each occupied voxel in `synapse_counts`, find the `closest_count` number
    of virtual fibers to it

    Returns:
        dict(tuple(i, j, k) voxel -> idx into virtual_fibers
    '''
    L.debug('closest_fibers_per_voxel...')
    ijks = np.transpose(np.nonzero(synapse_counts.raw))
    pos = synapse_counts.indices_to_positions(ijks)
    pos += synapse_counts.voxel_dimensions / 2.
    distances = calc_distances(pos, virtual_fibers[XYZUVW].values)

    closest_count = min(closest_count, distances.shape[1] - 1)

    # get closest_count closest virtual fibers
    partition = np.argpartition(distances, closest_count, axis=1)[:, :closest_count]
    fibers = pd.DataFrame(virtual_fibers.index[partition].values)
    return pd.concat([fibers, pd.DataFrame(ijks, columns=IJK)], axis=1)


def calc_distances_vectorized(candidates, virtual_fibers):
    '''For every synapse compute the distance to each candidate fiber'''
    idx = candidates.loc[:, FIBER_COLS].fillna(0).values.astype(int)
    fiber_coord = virtual_fibers[XYZUVW].values[idx]
    starts = fiber_coord[:, :, VF_STARTS]
    directions = fiber_coord[:, :, VF_DIRS]
    synapse_position = candidates.loc[:, XYZ].values
    distance_to_start = (synapse_position - starts.transpose(1, 0, 2)).transpose(1, 0, 2)
    return np.linalg.norm(np.cross(distance_to_start, directions), axis=2)


def assign_synapse_fiber(candidates, virtual_fibers, sigma):
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
    sgids = candidates.loc[:, FIBER_COLS].values[np.arange(len(idx)), idx]
    return pd.DataFrame({'sgid': sgids})
