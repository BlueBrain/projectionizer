'''Data and geometry related to somatosensory cortex'''
import logging
import os

import numpy as np
import pandas as pd
import voxcell

from projectionizer.utils import mask_by_region, XYZUVW, load

L = logging.getLogger(__name__)

REGION_INFO = {'s1hl': {'region': 'primary somatosensory cortex, hindlimb region',
                        'layer6': 'primary somatosensory cortex, hindlimb region, layer 6'},
               's1': {'region': [725, 726, 728, 730, ],
                      'layer6': [1124, 1130, 1142, 1148, ], },
               }


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


def relative_distance_layer(distance, layer_ph):
    '''
    Get the relative voxel distance in a layer from the bottom of the layer to the top.

    Args:
        distance (VoxelData): distance from the bottom
        layer_ph (VoxelData): PH for the layer ([PH]<layer>.nrrd)

    Returns:
        relative_distance (VoxelData): relative_voxel distance
    '''
    top = layer_ph[..., 1]
    bottom = layer_ph[..., 0]
    thickness = top - bottom
    relative_height = distance.raw - bottom

    return distance.with_data(relative_height / thickness)


def recipe_to_relative_heights_per_layer(distance, layers, voxel_path):
    '''
    Get the relative voxel height in a layer from the bottom of the layer to the top.

    Args:
        distance (VoxelData): distance from the bottom
        layers(list of tuples(name, thickness)): aranged from 'bottom' to 'top'
        voxel_path (str): path to the atlas

    Returns:
        relative_height (VoxelData): relative voxel heights in <layer_index>.<fraction> format

    TODO: add tests and test data
    '''
    relative_heights = np.full_like(distance.raw, np.nan)
    names, _ = zip(*layers)

    for layer_index, layer in enumerate(names):
        ph = load(os.path.join(voxel_path, '[PH]{}.nrrd'.format(layer)))
        mask = (ph.raw[..., 0] < distance.raw) & (distance.raw < ph.raw[..., 1])
        ph.raw[np.invert(mask), :] = np.nan
        relative_height = relative_distance_layer(distance, ph.raw)
        idx = np.isfinite(relative_height.raw)  # pylint: disable=assignment-from-no-return

        # Combine the layer_index and the voxels' relative heights in that layer.
        relative_heights[idx] = layer_index + relative_height.raw[idx]

    return distance.with_data(relative_heights)


def recipe_to_relative_height_and_density(layers,
                                          low_layer,
                                          low_fraction,
                                          high_layer,
                                          high_fraction,
                                          distribution):
    '''Convert recipe style layer & density values to relative height and density values.

    Relative height is relative height in a layer. E.g., relative heights 0.50 and 2.60 mean
    50% of the bottom layer and 60% of the third layer from the bottom respectively.

    Args:
        layers(list of tuples(name, thickness)): aranged from 'bottom' to 'top'
        low_layer(str): layer 'name'
        low_fraction(float): Fraction into low_layer from which to start the region
        high_layer(str): layer 'name' (1..6)
        high_fraction(float): Fraction into high_layer from which to end the region
        distribution(iter of tuples: (percent, density synapses/um3): density is assigned
        to each portion of the region: percent is the midpoint 'histogram'

    Return:
        list of lists of [relative_height, synapse density]
    '''
    layer_names, _ = zip(*layers)
    distribution = np.array(distribution)
    low_ind, high_ind = layer_names.index(low_layer), layer_names.index(high_layer)
    height_diff = high_ind + high_fraction - low_ind - low_fraction
    heights = distribution[:, 0]
    heights = np.hstack((0, 0.5 * (heights[:-1] + heights[1:]), 1))

    # Relative height in the <layer_index>.<fraction> format
    relative_height = low_ind + low_fraction + heights * height_diff
    distribution = np.hstack((distribution[:, 1], distribution[-1, 1]))

    return np.vstack((relative_height, distribution)).T.tolist()


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
