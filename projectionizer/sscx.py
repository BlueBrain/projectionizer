'''Data and geometry related to somatosensory cortex'''
import logging
import os

import numpy as np
import pandas as pd
import voxcell

from projectionizer.utils import mask_by_region, XYZUVW

L = logging.getLogger(__name__)

REGION_INFO = {'s1hl': {'region': 'primary somatosensory cortex, hindlimb region',
                        'layer6': 'primary somatosensory cortex, hindlimb region, layer 6'},
               's1': {'region': [725, 726, 728, 730, ],
                      'layer6': [1124, 1130, 1142, 1148, ], },
               'CA3_CA1': {'region': [725, 726, 728, 730, ],
                           'layer6': [1124, 1130, 1142, 1148, ], }}


def column_layers(layers):
    '''

    ie: for the SSCX layer 6 is at the bottom

    Args:
        layers(list of tuples): of (name, thickness), arranged from 'bottom' to 'top'

    Returns:
        layer_starts, layer_thickness: dictionaries of LayerName -> starts/thickness
    '''
    names, thickness = zip(*layers)
    cum_thickness = np.cumsum([0] + list(thickness))
    layer_starts = dict(zip(names, cum_thickness))
    return layer_starts, dict(layers)


def recipe_to_height_and_density(layers,
                                 low_layer,
                                 low_fraction,
                                 high_layer,
                                 high_fraction,
                                 distribution):
    '''Convert recipe style layer & density values to absolute height & density values

    Args:
        layers(list of tuples(name, thickness)): aranged from 'bottom' to 'top'
        low_layer(str): layer 'name'
        low_fraction(float): Fraction into low_layer from which to start the region
        high_layer(str): layer 'name' (1..6)
        high_fraction(float): Fraction into high_layer from which to end the region
        distribution(iter of tuples: (percent, density synapses/um3): density is assigned
        to each portion of the region: percent is the midpoint 'histogram'

    Return:
        list of tuples of (absolute height, synapse density)
    '''
    layer_starts, layer_thickness = column_layers(layers)

    distribution = np.array(distribution)
    heights = distribution[:, 0]
    heights = np.hstack((0, 0.5 * (heights[0:-1] + heights[1:]), 1))

    density = distribution[:, 1]
    density = np.hstack((density, density[-1]))

    bottom = layer_starts[low_layer] + low_fraction * layer_thickness[low_layer]
    top = layer_starts[high_layer] + high_fraction * layer_thickness[high_layer]
    diff = top - bottom
    return [(bottom + diff * low, density)
            for low, density in zip(heights, density)]


def load_s1_virtual_fibers(geometry, voxel_path, prefix):
    '''get the s1 virtual fibers

    One is 'created' for every voxel that is in layer 6 and has distance 0
    '''
    prefix = prefix or ''
    layer6_region = REGION_INFO[geometry]['layer6']
    mask = mask_by_region(layer6_region, voxel_path, prefix)
    distance = voxcell.VoxelData.load_nrrd(os.path.join(voxel_path, prefix + 'distance.nrrd'))
    distance.raw[np.invert(mask)] = np.nan
    idx = np.transpose(np.nonzero(distance.raw == 0.0))
    fiber_pos = distance.indices_to_positions(idx)

    count = None  # should be a parameter
    if count is not None:
        fiber_pos = fiber_pos[np.random.choice(np.arange(len(fiber_pos)), count)]

    orientation = voxcell.OrientationField.load_nrrd(
        os.path.join(voxel_path, prefix + 'orientation.nrrd'))
    orientation.raw = orientation.raw.astype(np.int8)
    y_vec = np.array([0, 1, 0])
    fiber_directions = -y_vec.dot(orientation.lookup(fiber_pos))

    return pd.DataFrame(np.hstack((fiber_pos, fiber_directions)), columns=XYZUVW)
