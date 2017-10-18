'''Data and geometry related to somatosensory cortex'''

import numpy as np


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


def recipe_to_height_and_density(low_layer,
                                 low_fraction,
                                 high_layer,
                                 high_fraction,
                                 distribution,
                                 mult=1.0):
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
    return [(bottom + diff * low, mult*density)
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
# synaptic parameters are mainly derived from Amitai, 1997; Castro-Alamancos & Connors 1997; Gil et al. 1997; Bannister et al. 2010 SR-->
def get_gamma_parameters(mn, sd):
    '''from projection_utility.py'''
    return ((mn / sd)**2, (sd**2) / mn)  # k, theta or shape, scale


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
    # (https://bbpteam.epfl.ch/project/issues/browse/NSETM-256?focusedCommentId=56509&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-56509)
    synapse_data[:, SynapseColumns.DELAY] = 0.25

    synapse_data[:, SynapseColumns.ISEC] = synapses[:, 2]
    synapse_data[:, SynapseColumns.IPT] = synapses[:, 3]
    synapse_data[:, SynapseColumns.OFFSET] = synapses[:, 4]

    def gamma(param):
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
