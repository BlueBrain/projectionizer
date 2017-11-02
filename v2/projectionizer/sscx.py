'''Data and geometry related to somatosensory cortex'''

import numpy as np
import numpy.matlib
from scipy.stats import norm  # pylint: disable=no-name-in-module

LAYERS = ('1', '2', '3', '4', '5', '6', )
# layer thickness from recipe
LAYER_THICKNESS = np.array([164.94915873, 148.87602025, 352.92508322,
                            189.57183895, 525.05585701, 700.37845971, ])
LAYER_BOUNDARIES = np.cumsum(list(reversed(LAYER_THICKNESS)))
LAYER_STARTS = {'6': 0,
                '5': LAYER_BOUNDARIES[0],
                '4': LAYER_BOUNDARIES[1],
                '3': LAYER_BOUNDARIES[2],
                '2': LAYER_BOUNDARIES[3],
                '1': LAYER_BOUNDARIES[4],
                }
LAYER_THICKNESS = {name: float(thickness) for name, thickness in
                   zip(LAYERS, LAYER_THICKNESS)}

CLOSEST_COUNT = 25
EXCLUSION = 120  # 60 # 3 times std?


def recipe_to_height_and_density(low_layer,
                                 low_fraction,
                                 high_layer,
                                 high_fraction,
                                 distribution):
    '''Convert recipe style layer & density values to absolute height & density values

    Args:
        low_layer(str): layer 'name'
        low_fraction(float): Fraction into low_layer from which to start the region
        high_layer(str): layer 'name' (1..6)
        high_fraction(float): Fraction into high_layer from which to end the region
        distribution(iter of tuples: (percent, density synapses/um3): density is assigned
        to each portion of the region: percent is the midpoint 'histogram'
        mult(float): multiply the densities by this

    Return:
        list of tuples of (absolute height, synapse density)
    '''
    distribution = np.array(distribution)
    heights = distribution[:, 0]
    heights = np.hstack((0, 0.5 * (heights[0:-1] + heights[1:]), 1))

    density = distribution[:, 1]
    density = np.hstack((density, density[-1]))

    bottom = LAYER_STARTS[low_layer] + low_fraction * LAYER_THICKNESS[low_layer]
    top = LAYER_STARTS[high_layer] + high_fraction * LAYER_THICKNESS[high_layer]
    diff = top - bottom
    return [(bottom + diff * low, density)
            for low, density in zip(heights, density)]


class SynapseColumns(object):
    '''columns index in nrn.h5 style file'''
    SGID = 0
    DELAY = 1

    ISEC = 2
    IPT = 3
    OFFSET = 4

    WEIGHT = 8  # G / CONDUCTANCE
    U = 9
    D = 10
    F = 11
    DTC = 12
    SYNTYPE = 13


# from thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6_specific.xml
# synaptic parameters are mainly derived from Amitai, 1997;
# Castro-Alamancos & Connors 1997; Gil et al. 1997; Bannister et al. 2010 SR
def get_gamma_parameters(mn, sd):
    '''from projection_utility.py'''
    return ((mn / sd) ** 2, (sd ** 2) / mn)  # k, theta or shape, scale


SYNAPSE_PARAMS = {
    'id': 120,
    # peak synaptic conductance for generalized e-e after PSP scaling 0.792+-0.528 EM
    'gsyn': get_gamma_parameters(0.792, 0.528),
    'Use': get_gamma_parameters(0.75, 0.02),  # Analogous to transmitter release probability SR -->
    'D': get_gamma_parameters(671, 17),  # Time constant for recovery from depression SR -->
    'F': get_gamma_parameters(17, 5),  # Time constant for recovery from facilitation SR -->
    'DTC': get_gamma_parameters(1.74, 0.2),  # decay time constant SR -->
    # Absolute synaptic efficacy - not used, but a placeholder continuing from legacy nrn.h5 SR -->
    'Ase': get_gamma_parameters(1, 0.01),
}


def create_synapse_data(synapses):
    '''return numpy array for `synapses` with the correct parameters

    Args:
        synapses(np.array): columns - 1 - source gid
                                      2 - section_id
                                      3 - segment_id
                                      4 - location along segment (in um)
    '''
    synapse_data = np.zeros((len(synapses), 19), dtype=np.float)
    synapses = np.array(synapses)

    synapse_data[:, SynapseColumns.SGID] = synapses[:, 1]

    # TODO XXX:calculate from y-pos?
    # Note: this has to be > 0
    # (https://bbpteam.epfl.ch/project/issues/browse/NSETM-256?focusedCommentId=56509)
    synapse_data[:, SynapseColumns.DELAY] = 0.25

    synapse_data[:, SynapseColumns.ISEC] = synapses[:, 2]
    synapse_data[:, SynapseColumns.IPT] = synapses[:, 3]
    synapse_data[:, SynapseColumns.OFFSET] = synapses[:, 4]

    def gamma(param):
        '''given `param`, look it up in SYNAPSE_PARAMS, return random pulls from gamma dist '''
        return np.random.gamma(shape=SYNAPSE_PARAMS[param][0],
                               scale=SYNAPSE_PARAMS[param][1],
                               size=len(synapses))

    synapse_data[:, SynapseColumns.WEIGHT] = gamma('gsyn')
    synapse_data[:, SynapseColumns.U] = gamma('Use')
    synapse_data[:, SynapseColumns.D] = gamma('D')
    synapse_data[:, SynapseColumns.F] = gamma('F')
    synapse_data[:, SynapseColumns.DTC] = gamma('DTC')
    synapse_data[:, SynapseColumns.SYNTYPE] = SYNAPSE_PARAMS['id']

    return synapse_data


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
    locations = numpy.matlib.repmat(locations, virtual_fiber_count, 1)

    distances = np.linalg.norm(np.cross((locations - starts), directions), axis=1)

    distances = distances.reshape(virtual_fiber_count, locations_count)
    return distances.T


def get_voxelized_fiber_distances(synapse_counts,
                                  virtual_fibers,
                                  closest_count=CLOSEST_COUNT,
                                  exclusion=EXCLUSION):
    '''for each occupied voxel in `synapse_counts`, find the `closest_count` number
    of virtual fibers to it

    Returns:
        dict(tuple(i, j, k) voxel -> idx into virtual_fibers
    '''
    ijks = np.transpose(np.nonzero(synapse_counts.raw))
    pos = synapse_counts.indices_to_positions(ijks)
    pos += synapse_counts.voxel_dimensions / 2.
    distances = calc_distances(pos, virtual_fibers)

    # shortcut: exclusion defines a cube around the point, the distance can't be
    # more than the sqrt(2) from that
    distances[1.41 * exclusion < distances] = np.nan

    # check if there are intersections w/ the virtual_fibers and occupied voxels
    # np.count_nonzero(np.any(np.invert(np.isnan(distances)), axis=1))

    closest_count = min(closest_count, distances.shape[1] - 1)

    # get closest_count closest virtual fibers
    partition = np.argpartition(distances, closest_count, axis=1)[:, :(closest_count + 1)]
    ret = {tuple(ijk): p for ijk, p in zip(ijks, partition)}
    return ret


# from recipe/Projection_Recipes/Thalamocortical_VPM/
# /thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn*.xml
SIGMA = 20.


def assign_synapse_fiber(locations,
                         synapse_counts,
                         virtual_fibers,
                         voxelized_fiber_distances,
                         sigma=SIGMA,
                         closest_count=CLOSEST_COUNT):
    '''

    Args:
        locations(np.arraya of Nx3): xyz positions of synapses
        synapse_counts(VoxelData): voxels occupied by synapses
        virtual_fibers(np.array Nx6): point and direction vectors of virtual_fibers
        voxelized_fiber_distances(dict tuple(ijk) -> idx of closest virtual fibers):
        fast lookup for distances, computed with get_voxelized_fiber_distances
        sigma(float): used for normal distribution
    '''
    default = np.zeros(closest_count)
    fiber_idx = [voxelized_fiber_distances.get(tuple(ijk), default)
                 for ijk in synapse_counts.positions_to_indices(locations)]
    fiber_idx = np.vstack(fiber_idx).astype(np.int)

    fibers = []
    for loc, fidx in zip(locations, fiber_idx):
        distances = calc_distances(loc[np.newaxis], virtual_fibers[fidx, :])
        # want to choose the 'best' one based on a normal distribution based on distance
        prob = norm.pdf(distances, 0, sigma)
        prob = np.nan_to_num(prob)

        idx = choice(prob)
        fibers.append(fidx[idx[0]])

    return fibers


def mask_far_fibers(fibers, origin, exclusion_box):
    """Mask fibers outside of exclusion_box centered on origin"""
    fibers = np.rollaxis(fibers, 1)
    fibers = np.abs(fibers - origin) < exclusion_box
    return np.all(fibers, axis=2).T


def choice(probabilities):
    '''manually sample one of them: would use np.random.choice, but that's only the 1d case
    '''
    cum_distances = np.cumsum(probabilities, axis=1)
    cum_distances = cum_distances / np.sum(probabilities, axis=1, keepdims=True)
    rand_cutoff = np.random.random((len(cum_distances), 1))
    idx = np.argmax(rand_cutoff < cum_distances, axis=1)
    return idx
