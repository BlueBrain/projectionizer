'''Data and geometry related to somatosensory cortex'''
import logging
import os

import numpy as np
import pandas as pd
import voxcell
from voxcell.nexus.voxelbrain import Atlas
import scipy.ndimage as nd

from projectionizer.utils import mask_by_region, XYZUVW, load

L = logging.getLogger(__name__)

REGION_INFO = {'s1hl': {'region': 'primary somatosensory cortex, hindlimb region',
                        'layer3': 'primary somatosensory cortex, hindlimb region, layer 3',
                        'layer4': 'primary somatosensory cortex, hindlimb region, layer 4',
                        'layer5': 'primary somatosensory cortex, hindlimb region, layer 5',
                        'layer6': 'primary somatosensory cortex, hindlimb region, layer 6'},
               's1': {'region': [725, 726, 728, 730, ],
                      'layer3': [1121, 1127, 1139, 1145, ],
                      'layer4': [1122, 1128, 1140, 1146, ],
                      'layer5': [1123, 1129, 1141, 1147, ],
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


def get_fiber_positions(distance, geometry, voxel_path, prefix):
    '''Get the fiber positions.

    i.e., the positions of the voxels that lie on the border of L5 and L3/L4.
    '''
    layer5_region = REGION_INFO[geometry]['layer5']
    layer4_region = REGION_INFO[geometry]['layer4']
    layer3_region = REGION_INFO[geometry]['layer3']
    mask5 = mask_by_region(layer5_region, voxel_path, prefix)
    mask4 = mask_by_region(layer4_region, voxel_path, prefix)
    mask3 = mask_by_region(layer3_region, voxel_path, prefix)

    bd = nd.binary_dilation(mask4 | mask3)
    mask = mask5 & bd
    idx = np.transpose(np.where(mask))

    return distance.indices_to_positions(idx)


def get_fiber_directions(fiber_pos, atlas, prefix):
    '''Get the fiber directions at fiber_pos positions.
    '''
    orientation = atlas.load_data(prefix + 'orientation', cls=voxcell.OrientationField)
    orientation.raw = orientation.raw.astype(np.int8)
    y_vec = np.array([0, 1, 0])
    R = orientation.lookup(fiber_pos)

    return np.matmul(R, y_vec)


def mask_layer_6_bottom(distance, geometry, voxel_path, prefix):
    '''Get the mask for the bottom of layer6.'''
    layer6_region = REGION_INFO[geometry]['layer6']
    mask6 = mask_by_region(layer6_region, voxel_path, prefix)
    distance.raw[np.invert(mask6)] = np.nan
    min_dist = np.min(distance.raw[np.isfinite(distance.raw)])

    return distance.raw == min_dist


def load_s1_virtual_fibers(geometry, voxel_path, prefix):
    '''get the s1 virtual fibers

    Tracing back from L5/L43 boundary along the orientations to bottom of L6 and picking
    those fibers that do hit the bottom of L6 voxels.
    '''
    prefix = prefix or ''

    atlas = Atlas.open(voxel_path)
    distance = atlas.load_data(prefix + 'distance')
    fiber_pos = get_fiber_positions(distance, geometry, voxel_path, prefix)
    mask = mask_layer_6_bottom(distance, geometry, voxel_path, prefix)

    count = None  # should be a parameter
    if count is not None:
        fiber_pos = fiber_pos[np.random.choice(np.arange(len(fiber_pos)), count)]

    fiber_dir = get_fiber_directions(fiber_pos, atlas, prefix)

    ret = []

    # TODO: more effective way of tracing the fibers to bottom of L6.
    for pos, dirs in zip(fiber_pos, fiber_dir):
        try:
            while not mask[tuple(distance.positions_to_indices(pos))]:
                pos -= dirs * distance.voxel_dimensions[1]
        except voxcell.exceptions.VoxcellError:
            # Expecting out of bounds error
            continue

        ret.append(np.concatenate((pos, dirs)))

    ret = np.vstack(ret)

    return pd.DataFrame(ret, columns=XYZUVW)
