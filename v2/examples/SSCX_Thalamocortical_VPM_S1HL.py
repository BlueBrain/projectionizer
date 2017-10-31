'''Create projections for S1HL circuit'''
import json
import os

from functools import partial

import numpy as np
import pandas as pd

import voxcell
from voxcell import build
from bluepy.v2.circuit import Circuit
from bluepy.v2.enums import Cell, Section, Segment
from neurom import NeuriteType

from decorators import timeit

import map_parallelize
from projectionizer import projection, sscx, utils


BASE_CIRCUIT = '/gpfs/bbp.cscs.ch/project/proj64/circuits/S1HL/20171004/'

PREFIX = 'C63CB79F-392A-4873-9949-0D347682253A-'
VOXEL_PATH = os.path.join(BASE_CIRCUIT, '.atlas/')

'''
height: dataset gives total thickness along normal
orientation: dataset gives rotation quaternion to align morphology Y-axis along normal
distance: distance along normal to bottom of Layer 6
'''

# regions of interest for the S1HL
REGION_NAME = 'primary somatosensory cortex, hindlimb region'
LAYER6_NAME = 'primary somatosensory cortex, hindlimb region, layer 6'


def mask_by_region(region_name, path, prefix):
    '''
    Args:
        region_name(str): name to look up in atlas
        path(str): path to where nrrd files are, must include 'brain_regions.nrrd'
        prefix(str): Prefix (ie: uuid) used to identify atlas/voxel set
    '''
    atlas = voxcell.VoxelData.load_nrrd(os.path.join(path, prefix + 'brain_regions.nrrd'))
    with open(os.path.join(path, prefix[:-1] + '.json')) as fd:
        hierarchy = voxcell.Hierarchy(json.load(fd))
    mask = build.mask_by_region_names(atlas.raw, hierarchy, [region_name])
    return mask


def get_virtual_fibers(layer_name=LAYER6_NAME, count=None, path=VOXEL_PATH, prefix=PREFIX):
    '''return the fiber positions and direction vectors

    Args:
        layer_name(str): layer name to look up in atlas
        count(str): maximum number of virtual fibers to return
        path(str): path to where nrrd files are, must include 'brain_regions.nrrd'
        prefix(str): Prefix (ie: uuid) used to identify atlas/voxel set

    Returns:
        a Nx6 matrix, w/ 0:3 being the start positions, and 3:6 being the direction vector

    '''
    mask = mask_by_region(layer_name, path, prefix)
    distance_path = os.path.join(path, prefix + 'distance.nrrd')
    distance = voxcell.VoxelData.load_nrrd(distance_path)
    distance.raw[np.invert(mask)] = np.nan
    idx = np.transpose(np.nonzero(distance.raw == 0.0))
    fiber_pos = distance.indices_to_positions(idx)

    if count is not None:
        fiber_pos = fiber_pos[np.random.choice(np.arange(len(fiber_pos)), count)]

    orientation_path = os.path.join(path, prefix + 'orientation.nrrd')
    orientation = voxcell.OrientationField.load_nrrd(orientation_path)
    orientation.raw = orientation.raw.astype(np.int8)
    orientations = orientation.lookup(fiber_pos)
    y_vec = np.array([0, 1, 0])
    fiber_directions = -y_vec.dot(orientations)

    return np.hstack((fiber_pos, fiber_directions))


def get_heights(region_name=REGION_NAME, path=VOXEL_PATH, prefix=PREFIX):
    '''return a VoxelData instance w/ all the heights for given region_name

    distance is defined as from the voxel to the bottom of L6, voxels
    outside of region_name are set to 0

    Args:
        region_name(str): name to look up in atlas
        path(str): path to where nrrd files are, must include 'brain_regions.nrrd'
        prefix(str): Prefix (ie: uuid) used to identify atlas/voxel set
    '''
    mask = mask_by_region(region_name, path, prefix)
    distance = voxcell.VoxelData.load_nrrd(os.path.join(path, prefix + 'distance.nrrd'))
    distance.raw[np.invert(mask)] = 0.
    return distance


def build_voxel_synapse_count(height, distmap, path=VOXEL_PATH, prefix=PREFIX):
    raw = np.zeros_like(height.raw, dtype=np.uint)

    voxel_volume = np.prod(np.abs(height.voxel_dimensions))
    for dist in distmap:
        for (bottom, density), (top, _) in zip(dist[:-1], dist[1:]):
            idx = np.nonzero((bottom < height.raw) & (height.raw < top))
            raw[idx] = int(voxel_volume * density)

    return height.with_data(raw)


def _pick_syns(args, circuit):
    '''function to use map_parallelize'''
    min_xyz, max_xyz, count = args
    return projection.pick_synapses_voxel(
        circuit, min_xyz, max_xyz, count, segment_pref=utils.segment_pref)


def pick_synapses(circuit, synapse_counts, distmap, map_=map):
    def get_xyz_counts():
        idx = np.nonzero(synapse_counts.raw)

        min_xyzs = synapse_counts.indices_to_positions(np.transpose(idx))
        max_xyzs = min_xyzs + synapse_counts.voxel_dimensions

        for min_xyz, max_xyz, count in zip(min_xyzs, max_xyzs, synapse_counts.raw[idx]):
            yield min_xyz, max_xyz, count

    xyz_counts = list(get_xyz_counts())
    ps = partial(_pick_syns, circuit=circuit)
    synapses = map_(ps, xyz_counts)
    return synapses


@timeit('Pick Segments')
def sample_synapses(circuit, map_):
    #XXX: sampling takes a long time, use this to load faster
    synapses = pd.read_feather('/gpfs/bbp.cscs.ch/project/proj30/mgevaert/S1HL_20171004.feather')

    from SSCX_Thalamocortical_VPM_hex import y_distmap_3_4, y_distmap_5_6
    distmap = [sscx.recipe_to_height_and_density('4', 0, '3', 0.5, y_distmap_3_4, mult=2.6),
               sscx.recipe_to_height_and_density('6', 0.85, '5', 0.6, y_distmap_5_6, mult=2.6), ]
    heights = get_heights()
    synapse_counts = build_voxel_synapse_count(heights, distmap)
    #synapses = pick_synapses(circuit, synapse_counts, distmap, map_=map_)
    return synapses, synapse_counts


@timeit('Assign vector virtual fibers')
def assign_synapses_vector_fibers(synapses, synapse_counts, map_):
    synapses.rename(columns={'gid': 'tgid'}, inplace=True)

    virtual_fibers = get_virtual_fibers()
    voxelized_fiber_distances = sscx.get_voxelized_fiber_distances(synapse_counts, virtual_fibers)

    synapse_locations = synapses[list('xyz')].values

    asf = partial(sscx.assign_synapse_fiber,
                  synapse_counts=synapse_counts,
                  virtual_fibers=virtual_fibers,
                  voxelized_fiber_distances=voxelized_fiber_distances)
    CHUNK_SIZE = 100000
    fiber_id = map_(asf, toolz.itertoolz.partition_all(synapse_locations, CHUNK_SIZE))
    fiber_id = np.hstack(fiber_id)
    synapses['sgid'] = fiber_id
    return synapses


def main():
    base_circuit = BASE_CIRCUIT
    circuit = os.path.join(base_circuit, 'CircuitConfig')
    circuit = Circuit(circuit)
    synapses, synapse_counts = sample_synapses(circuit, map_=map_parallelize.map_parallelize)

    #reduce syn count to manageable fraction
    synapses = synapses.sample(frac=0.001)  # ~137529
    # synapses = synapses.sample(frac=0.01) # ~1375290

    assigned_synapses = assign_synapses_vector_fibers(synapses, synapse_counts, map_=map_parallelize.map_parallelize)

    from SSCX_Thalamocortical_VPM_hex import prune, write
    remaining_synapses = prune(assigned_synapses, circuit, parallelize=True)
    write(remaining_synapses, output)


# if __name__ == '__main__':
#    main()
