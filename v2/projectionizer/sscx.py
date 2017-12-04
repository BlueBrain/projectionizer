'''Data and geometry related to somatosensory cortex'''
import logging

import numpy as np


L = logging.getLogger(__name__)

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

EXCLUSION = 60  # 3 times std?

REGION_INFO = {'s1hl':
               {'region': 'primary somatosensory cortex, hindlimb region',
                'layer6': 'primary somatosensory cortex, hindlimb region, layer 6',
                },
               's1':
               {'region': [725, 726, 728, 730, ],
                'layer6': [1124, 1130, 1142, 1148, ],
                }
               }

y_distmap_3_4 = (
    (0.05, 0.01),
    (0.15, 0.02),
    (0.25, 0.03),
    (0.35, 0.04),
    (0.45, 0.04),
    (0.55, 0.04),
    (0.65, 0.03),
    (0.75, 0.02),
    (0.85, 0.01),
    (0.95, 0.01),
)
y_distmap_5_6 = (
    (0.05, 0.005),
    (0.15, 0.01),
    (0.25, 0.015),
    (0.35, 0.02),
    (0.45, 0.0225),
    (0.55, 0.025),
    (0.65, 0.0275),
    (0.75, 0.03),
    (0.85, 0.015),
    (0.95, 0.005),
)


def get_distmap():
    return [recipe_to_height_and_density('4', 0, '3', 0.5, y_distmap_3_4),
            recipe_to_height_and_density('6', 0.85, '5', 0.6, y_distmap_5_6), ]


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


def mask_far_fibers(fibers, origin, exclusion_box):
    """Mask fibers outside of exclusion_box centered on origin"""
    fibers = np.rollaxis(fibers, 1)
    fibers = np.abs(fibers - origin) < exclusion_box
    return np.all(fibers, axis=2).T
