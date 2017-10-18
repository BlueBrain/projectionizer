import json
import os
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
import voxcell
from bluepy.v2.circuit import Circuit
from bluepy.v2.enums import Cell, Section, Segment
from neurom import NeuriteType
from scipy.stats import norm

import map_parallelize
from projectionizer import projection, utils

matplotlib.use('Agg')



BUILD_PATH = '/gpfs/bbp.cscs.ch/project/proj64/circuits/S1HL/20171004/'

PREFIX = 'C63CB79F-392A-4873-9949-0D347682253A-'
VOXEL_PATH = os.path.join(BUILD_PATH, '.atlas/')
'''

height: dataset gives total thickness along normal
orientation: dataset gives rotation quaternion to align morphology Y-axis along normal
atlas = voxcell.VoxelData.load_nrrd(os.path.join(VOXEL_PATH, PREFIX + 'brain_regions.nrrd'))
distance = voxcell.VoxelData.load_nrrd(os.path.join(VOXEL_PATH, PREFIX + 'distance.nrrd'))
orientation = voxcell.OrientationField.load_nrrd(os.path.join(VOXEL_PATH, PREFIX + 'orientation.nrrd'))
#height = voxcell.VoxelData.load_nrrd(os.path.join(VOXEL_PATH, prefix + 'height.nrrd'))
with open(os.path.join(VOXEL_PATH, PREFIX[:-1] + '.json')) as fd:
    hierarchy = voxcell.Hierarchy(json.load(fd))
'''


CLOSEST_COUNT = 25
EXCLUSION = 120  # 60 # 3 times std?
SIGMA = 20

'''
'''


def iter_chunked_array(array, chunk_size):
    for pos in xrange(0, len(array), chunk_size):
        yield array[pos:pos + chunk_size]


def get_voxelized_fiber_distances(synapse_counts,
                                  virtual_fibers,
                                  closest_count=CLOSEST_COUNT,
                                  exclusion=EXCLUSION):
    '''for each occupied voxel in `synapse_counts`, find the `closest_count` number
    of virtual fibers to it

    Returns:
        dict(tuple(i, j, k) voxel -> idx into virtual_fibers
    '''
    ret = {}
    ijks = np.transpose(np.nonzero(synapse_counts.raw))
    pos = synapse_counts.indices_to_positions(ijks)
    pos += synapse_counts.voxel_dimensions / 2.
    distances = calc_distances(pos, virtual_fibers)

    # shortcut: exclustion defines a cube around the point, the distance can't be
    # more than the sqrt(2) from that
    distances[1.41 * exclusion < distances] = np.nan

    # check if there are intersections w/ the virtual_fibers and occupied voxels
    # np.count_nonzero(np.any(np.invert(np.isnan(distances)), axis=1))

    closest_count = min(closest_count, distances.shape[1] - 1)

    # get closest_count closest minicolumns
    partition = np.argpartition(distances, closest_count, axis=1)[:, :closest_count]
    ret = {tuple(ijk): p for ijk, p in zip(ijks, partition)}
    return ret


def assign_synapse_fiber(locations, synapse_counts, virtual_fibers, voxelized_fiber_distances):
    default = np.zeros(25)
    fiber_idx = [voxelized_fiber_distances.get(tuple(ijk), default)
                 for ijk in synapse_counts.positions_to_indices(locations)]
    fiber_idx = np.vstack(fiber_idx).astype(np.int)

    fibers = []
    for loc, fidx in zip(locations, fiber_idx):
        distances = calc_distances(loc[np.newaxis], virtual_fibers[fidx, :])
        # want to choose the 'best' one based on a normal distribution based on distance
        prob = norm.pdf(distances, 0, SIGMA)
        prob = np.nan_to_num(prob)

        # manually sample one of them: would use np.random.choice, but that's only the 1d case
        cum_distances = np.cumsum(prob, axis=1)
        cum_distances = cum_distances / np.sum(prob, axis=1, keepdims=True)
        rand_cutoff = np.random.random((len(cum_distances), 1))
        idx = np.argmax(rand_cutoff < cum_distances, axis=1)
        fibers.append(fidx[idx[0]])

    return fibers


def test_calc_distances():
    locations = np.array([[0,  0,  0],
                          [10,  0,  0],
                          [10, 10, 10],
                          [10, 10,  0]], dtype=np.float)
    virtual_fibers = np.array([[0.,  0.,  0.,  1.,  0.,  0.],
                               [0.,  0.,  0.,  0.,  1.,  0.],
                               [0.,  0.,  0.,  0.,  0.,  1.]])
    ret = calc_distances(locations, virtual_fibers)
    eq_(ret.shape, (4, 3))


def calc_distances(locations, virtual_fibers):
    '''find closest point from locations to fibers

    virtual_fibers is a Nx6 matrix, w/ 0:3 being the start positions, and 3:6 being the direction vector
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


def mask_by_region(region_name, path, prefix):
    from voxcell import build
    atlas = voxcell.VoxelData.load_nrrd(os.path.join(path, prefix + 'brain_regions.nrrd'))
    with open(os.path.join(path, prefix[:-1] + '.json')) as fd:
        hierarchy = voxcell.Hierarchy(json.load(fd))
    mask = build.mask_by_region_names(atlas.raw, hierarchy, [region_name])
    return mask


#LAYER_NAME = 'primary somatosensory cortex, layer 6'
LAYER6_NAME = 'primary somatosensory cortex, hindlimb region, layer 6'


def get_virtual_fibers(layer_name=LAYER6_NAME, count=None, path=VOXEL_PATH, prefix=PREFIX):
    '''return the fiber positions, based'''
    mask = mask_by_region(layer_name, path, prefix)
    path = os.path.join(path, prefix + 'distance.nrrd')
    distance = voxcell.VoxelData.load_nrrd(path)
    distance.raw[np.invert(mask)] = np.nan
    idx = np.transpose(np.nonzero(distance.raw == 0.0))
    fiber_pos = distance.indices_to_positions(idx)

    if count is not None:
        fiber_pos = fiber_pos[np.random.choice(np.arange(len(fiber_pos)), count)]

    path = os.path.join(path, prefix + 'orientation.nrrd')
    orientation = voxcell.OrientationField.load_nrrd(path)
    orientation.raw = orientation.raw.astype(np.int8)
    orientations = orientation.lookup(fiber_pos)
    y_vec = np.array([0, 1, 0])
    fiber_directions = -y_vec.dot(orientations)

    return np.hstack((fiber_pos, fiber_directions))


REGION_NAME = 'primary somatosensory cortex, hindlimb region'


def get_distances(region_name=REGION_NAME, path=VOXEL_PATH, prefix=PREFIX):
    '''return a VoxelData instance w/ all the distances for given region_name

    distance is defined as from the voxel to the bottom of L6
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
            print bottom, top, len(idx[0])
            raw[idx] = int(voxel_volume * density)

    return height.with_data(raw)


def plot_voxel(ax, idx, uvw=None, c='b'):
    '''one location per cell'''
    voxels = np.zeros(shape=np.max(idx, axis=0) + 1, dtype=bool)
    voxels[tuple(idx.T)] = True

    ax.voxels(voxels)
    if uvw is not None:
        ax.quiver(idx[:, 0], idx[:, 1], idx[:, 2],
                  uvw[:, 0], uvw[:, 1], uvw[:, 2],
                  facecolors=c)


def segment_pref(segs_df):
    '''don't want axons'''
    return (segs_df[Section.NEURITE_TYPE] != NeuriteType.axon).astype(float)


BASE_CIRCUIT = BUILD_PATH
base_circuit = BASE_CIRCUIT
circuit = os.path.join(base_circuit, 'CircuitConfig')
circuit = Circuit(circuit)


def _pick_syns(args):
    '''function to use map_parallelize'''
    min_xyz, max_xyz, count = args
    return projection.pick_synapses_voxel(
        circuit, min_xyz, max_xyz, count, segment_pref=segment_pref)


def pick_synapses(circuit, map_=map):
    def get_xyz_counts():
        height = get_distances()
        synapse_counts = build_voxel_synapse_count(height, distmap)
        idx = np.nonzero(synapse_counts.raw)

        min_xyzs = synapse_counts.indices_to_positions(np.transpose(idx))
        max_xyzs = min_xyzs + synapse_counts.voxel_dimensions

        for min_xyz, max_xyz, count in zip(min_xyzs, max_xyzs, synapse_counts.raw[idx]):
            yield min_xyz, max_xyz, count

    xyz_counts = list(get_xyz_counts())
    ps = partial(_pick_syns, circuit=circuit)
    synapses = map_(_pick_syns, xyz_counts)
    return synapses


def main():
    #heights = get_distances()
    #synapse_counts = build_voxel_synapse_count(heights, distmap)
    #synapses = pick_synapses(circuit, map_=map_parallelize.map_parallelize)

    synapses = load_feather('/gpfs/bbp.cscs.ch/project/proj30/mgevaert/S1HL_20171004.feather')

    virtual_fibers = get_virtual_fibers()
    voxelized_fiber_distances = get_voxelized_fiber_distances(synapse_counts, virtual_fibers)

    synapses = synapses.sample(frac=0.001)  # 137529
    # synapses = synapses.sample(frac=0.01) # 1375290
    synapse_locations = synapses[list('xyz')].values

    asf = partial(assign_synapse_fiber, synapse_counts=synapse_counts, virtual_fibers=virtual_fibers,
                  voxelized_fiber_distances=voxelized_fiber_distances)
    fiber_id = map_parallelize(asf, iter_chunked_array(synapse_locations, 100000))
    fiber_id = np.hstack(fiber_id)
    synapses['sgid'] = fiber_id

    synapses.rename(columns={'gid': 'tgid',
                             'Section.ID': 'section_id',
                             'Segment.ID': 'segment_id',
                             }, inplace=True)

# if __name__ == '__main__':
#    main()


'''
#dask pruning

import dask.dataframe as dd
from dask.distributed import Client
client = Client()

target_remove = 1.6/2.6*0.73/0.66 # =~ 0.6806526806526807
cutoff_means = _find_cutoff_means(circuit, synapses, target_remove=target_remove)

mtypes = circuit.cells.get(properties='mtype')
synapses = synapses.join(mtypes, on='tgid')
synapses.mtype.cat.remove_unused_categories(inplace=True)
df = dd.from_pandas(synapses, npartitions=16)
pathways = df.groupby(['sgid', 'tgid'])

cutoff_var = 1.0
def prune(df):
    if np.random.random() < norm.cdf(len(df), cutoff_means[df['mtype'].iloc[0]], cutoff_var):
        return df

asdf = pathways.apply(prune, meta=synapses).compute()
asdf.reset_index(drop=True, inplace=True)
'''
'''
l1_mask = build.mask_by_region_names(atlas.raw, hierarchy, ['primary somatosensory cortex, hindlimb region, layer 1'])
l1_locations = np.transpose(np.nonzero(l1_mask))
l1_loc = l1_locations[::5, :]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
plot_voxel(ax, fiber_idx, uvw)
plot_voxel(ax, l1_loc, c='r')
'''
