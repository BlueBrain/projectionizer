'''Utils for projectionizer'''
import json
import os
from itertools import chain
from multiprocessing import Pool
import signal

import numpy as np
import pandas as pd
from six import string_types
from voxcell import Hierarchy, VoxelData, build


X, Y, Z = 0, 1, 2
XYZUVW = list('xyzuvw')
IJK = list('ijk')
XYZ = list('xyz')


class ErrorCloseToZero(Exception):
    '''Raised if normalizing if sum of probabilities is close to zero'''


def write_feather(name, df):
    """Write a DataFrame to disk using feather serialization format

    Note: This performs destructive changes to the dataframe, caller must
    save it if they need an unchanged version
    """
    df.columns = map(str, df.columns)
    df = df.reset_index(drop=True)
    df.to_feather(name)


def load(filename):
    """Load a Pandas/Nrrd file based on the extension"""
    extension = os.path.splitext(filename)[1]
    try:
        return {
            '.feather': lambda: pd.read_feather(filename),
            '.nrrd': lambda: VoxelData.load_nrrd(filename),
            '.csv': lambda: pd.read_csv(filename, index_col=0),
            '.json': lambda: json.load(open(filename))
        }[extension]()
    except KeyError:
        raise NotImplementedError('Do not know how open: {}'.format(filename))


def load_all(inputs):
    '''load all `inputs`'''
    return [load(x.path) for x in inputs]


def map_parallelize(func, *it):
    '''apply func to all items in it, using a process pool

    Watch the memory usage!
    '''
    pool = Pool(14, lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    # map_async catches KeyboardInterrupt: https://stackoverflow.com/a/1408476/2533394
    res = pool.map_async(func, *it).get(9999999)  # pylint: disable=no-value-for-parameter
    pool.close()
    pool.join()
    return res


def normalize_probability(p):
    """ Normalize vector of probabilities `p` so that sum(p) == 1. """
    norm = np.sum(p)
    if norm < 1e-7:
        raise ErrorCloseToZero("Could not normalize almost-zero vector")
    return p / norm


def in_bounding_box(min_xyz, max_xyz, df):
    '''return boolean index of df rows that are in min_xyz/max_xyz

    df must have ['x', 'y', 'z'] columns
    '''
    ret = ((min_xyz[X] < df['x']) & (df['x'] < max_xyz[X]) &
           (min_xyz[Y] < df['y']) & (df['y'] < max_xyz[Y]) &
           (min_xyz[Z] < df['z']) & (df['z'] < max_xyz[Z]))
    return ret


def choice(probabilities):
    '''Given an array of shape (N, M) of probabilities (not necessarily normalized)
    returns an array of shape (N), with one element choosen from every rows according
    to the probabilities normalized on this row
    '''
    cum_distances = np.cumsum(probabilities, axis=1)
    cum_distances = cum_distances / np.sum(probabilities, axis=1, keepdims=True)
    rand_cutoff = np.random.random((len(cum_distances), 1))
    idx = np.argmax(rand_cutoff < cum_distances, axis=1)
    return idx


def mask_by_region(region, path, prefix):
    '''
    Args:
        region(str or list of region ids): name/ids to look up in atlas
        path(str): path to where nrrd files are, must include 'brain_regions.nrrd'
        prefix(str): Prefix (ie: uuid) used to identify atlas/voxel set
    '''
    atlas = VoxelData.load_nrrd(os.path.join(path, prefix + 'brain_regions.nrrd'))
    with open(os.path.join(path, 'hierarchy.json')) as fd:
        hierarchy = Hierarchy(json.load(fd))
    if isinstance(region, string_types):
        mask = build.mask_by_region_names(atlas.raw, hierarchy, [region])
    else:
        region_ids = list(chain.from_iterable(hierarchy.collect('id', id_, 'id')
                                              for id_ in region))

        mask = build.mask_by_region_ids(atlas.raw, region_ids)
    return mask
