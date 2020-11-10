'''Data and geometry related to somatosensory cortex'''
import logging

import numpy as np
import pandas as pd
import voxcell
import scipy.ndimage as nd

from projectionizer.utils import mask_by_region_ids, XYZUVW

L = logging.getLogger(__name__)


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


def recipe_to_relative_heights_per_layer(distance, atlas, layers):
    '''
    Get the relative voxel height in a layer from the bottom of the layer to the top.

    Args:
        distance (VoxelData): distance from the bottom
        atlas (voxcell.Atlas): atlas instance for the circuit
        layers(list of tuples(name, thickness)): aranged from 'bottom' to 'top'

    Returns:
        relative_height (VoxelData): relative voxel heights in <layer_index>.<fraction> format
    '''
    relative_heights = np.full_like(distance.raw, np.nan)
    names, _ = zip(*layers)

    for layer_index, layer in enumerate(names):
        ph = atlas.load_data('[PH]{}'.format(layer))
        mask = (ph.raw[..., 0] <= distance.raw) & (distance.raw <= ph.raw[..., 1])
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


def get_region_ids(atlas, layers, regions):
    '''Get region id's for the regions and layers.'''
    rmap = atlas.load_region_map()
    regex_str_regions = '@^({})$'.format('|'.join(regions))
    regex_str_layers = '@^.*({})$'.format('|'.join(layers))

    id_regions_children = rmap.find(regex_str_regions, attr='acronym', with_descendants=True)
    id_layers_all_regions = rmap.find(regex_str_layers, attr='acronym')
    id_wanted_layers = set.intersection(id_regions_children, id_layers_all_regions)

    return list(id_wanted_layers)


def mask_layers_in_regions(atlas, layers, regions):
    '''Get the mask for defined layers in all defined regions.'''
    brain_regions = atlas.load_data('brain_regions')
    ids = get_region_ids(atlas, layers, regions)

    return mask_by_region_ids(brain_regions.raw, ids)


def get_l5_l34_border_voxel_indices(atlas, regions):
    '''Get the fiber indices.

    i.e., the indices of the voxels that lie on the border of L5 and L3/L4.
    '''
    mask_l5 = mask_layers_in_regions(atlas, ['L5'], regions)
    mask_l34 = mask_layers_in_regions(atlas, ['L3', 'L4'], regions)
    mask = mask_l5 & nd.binary_dilation(mask_l34)

    return np.transpose(np.where(mask))


def get_fiber_directions(fiber_positions, atlas):
    '''Get the fiber directions at positions defined in fiber_positions.'''
    orientation = atlas.load_data('orientation', cls=voxcell.OrientationField)
    orientation.raw = orientation.raw.astype(np.int8)
    y_vec = np.array([0, 1, 0])
    R = orientation.lookup(fiber_positions)

    return np.matmul(R, y_vec)


def mask_layer_6_bottom(atlas, regions):
    '''Get the mask for the bottom of layer6.'''
    distance = atlas.load_data('[PH]y')
    mask6 = mask_layers_in_regions(atlas, ['L6'], regions)
    distance.raw[np.invert(mask6)] = np.nan
    min_dist = np.min(distance.raw[np.isfinite(distance.raw)])

    return distance.raw == min_dist


def ray_tracing(atlas, target_mask, fiber_positions, fiber_directions):
    '''Get virtual fiber start positions by ray_tracing.

    Args:
        atlas (voxcell.Atlas): atlas instance for the circuit
        target_mask (numpy.array): 3D array masking the potential startpoints (e.g. bottom of L4)
        fiber_positions (numpy.array): fiber positions to trace back from (e.g., L4/L5 boundary
                                       voxel positions)
        fiber_directions (numpy.array): directions of the fibers at the positions given at fiber_pos

    Return:
        (pandas.DataFrame): virtual fibers found
    '''
    ret = []
    distance = atlas.load_data('[PH]y')

    # TODO: more effective way of tracing the fibers to bottom of L6.
    for pos, dirs in zip(fiber_positions, fiber_directions):
        try:
            while not target_mask[tuple(distance.positions_to_indices(pos))]:
                pos -= dirs * distance.voxel_dimensions[1]
        except voxcell.exceptions.VoxcellError:
            # Expecting out of bounds error if ray tracing did not hit any voxels in mask
            continue

        ret.append(np.concatenate((pos, dirs)))

    ret = np.vstack(ret)

    return pd.DataFrame(ret, columns=XYZUVW)


def load_s1_virtual_fibers(atlas, regions):
    '''get the s1 virtual fibers

    Tracing back from L5/L43 boundary along the orientations to bottom of L6 and picking
    those fibers that do hit the bottom of L6 voxels.

    Args:
        atlas (voxcell.Atlas): atlas instance for the circuit
        regions (list): list of region acronyms

    Return:
        (pandas.DataFrame): virtual fibers found
    '''
    distance = atlas.load_data('[PH]y')
    mask = mask_layer_6_bottom(atlas, regions)
    fiber_pos = distance.indices_to_positions(get_l5_l34_border_voxel_indices(atlas, regions))

    count = None  # should be a parameter
    if count is not None:
        fiber_pos = fiber_pos[np.random.choice(np.arange(len(fiber_pos)), count)]

    fiber_dir = get_fiber_directions(fiber_pos, atlas)

    return ray_tracing(atlas, mask, fiber_pos, fiber_dir)
