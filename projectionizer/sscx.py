"""Data and geometry related to somatosensory cortex"""
import logging

import numpy as np

from projectionizer.utils import convert_layer_to_PH_format

L = logging.getLogger(__name__)


def relative_distance_layer(distance, layer_ph):
    """
    Get the relative voxel distance in a layer from the bottom of the layer to the top.

    Args:
        distance (VoxelData): distance from the bottom
        layer_ph (VoxelData): PH for the layer ([PH]<layer>.nrrd)

    Returns:
        relative_distance (VoxelData): relative_voxel distance
    """
    top = layer_ph[..., 1]
    bottom = layer_ph[..., 0]
    thickness = top - bottom
    relative_height = distance.raw - bottom

    return distance.with_data(relative_height / thickness)


def recipe_to_relative_heights_per_layer(distance, atlas, layers):
    """
    Get the relative voxel height in a layer from the bottom of the layer to the top.

    Args:
        distance (VoxelData): distance from the bottom
        atlas (voxcell.Atlas): atlas instance for the circuit
        layers(list of tuples(name, thickness)): aranged from 'bottom' to 'top'

    Returns:
        relative_height (VoxelData): relative voxel heights in <layer_index>.<fraction> format
    """
    relative_heights = np.full_like(distance.raw, np.nan)

    for layer_index, layer in enumerate(layers):
        ph = atlas.load_data("[PH]{}".format(convert_layer_to_PH_format(layer)))
        mask = (ph.raw[..., 0] <= distance.raw) & (distance.raw <= ph.raw[..., 1])
        ph.raw[np.invert(mask), :] = np.nan
        relative_height = relative_distance_layer(distance, ph.raw)
        idx = np.isfinite(relative_height.raw)  # pylint: disable=assignment-from-no-return

        # Combine the layer_index and the voxels' relative heights in that layer.
        relative_heights[idx] = layer_index + relative_height.raw[idx]

    return distance.with_data(relative_heights)


def recipe_to_relative_height_and_density(
    layers, low_layer, low_fraction, high_layer, high_fraction, distribution
):
    """Convert recipe style layer & density values to relative height and density values.

    Relative height is relative height in a layer. E.g., relative heights 0.50 and 2.60 mean
    50% of the bottom layer and 60% of the third layer from the bottom respectively.

    Args:
        layers(list): layer names arranged from 'bottom' to 'top'
        low_layer(str): low layer 'name'
        low_fraction(float): Fraction into low_layer from which to start the region
        high_layer(str): high layer 'name' (1..6)
        high_fraction(float): Fraction into high_layer from which to end the region
        distribution(iter of tuples: (percent, density synapses/um3): density is assigned
        to each portion of the region: percent is the midpoint 'histogram'

    Return:
        list of lists of [relative_height, synapse density]
    """
    distribution = np.array(distribution)
    low_ind, high_ind = layers.index(low_layer), layers.index(high_layer)
    height_diff = high_ind + high_fraction - low_ind - low_fraction
    heights = distribution[:, 0]
    heights = np.hstack((0, 0.5 * (heights[:-1] + heights[1:]), 1))

    # Relative height in the <layer_index>.<fraction> format
    relative_height = low_ind + low_fraction + heights * height_diff
    distribution = np.hstack((distribution[:, 1], distribution[-1, 1]))

    return np.vstack((relative_height, distribution)).T.tolist()
