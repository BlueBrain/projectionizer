'''Utils for projectionizer'''
from contextlib import contextmanager
import json
import os
from itertools import chain
import multiprocessing

import numpy as np
import pandas as pd
from six import string_types
from voxcell import Hierarchy, VoxelData
import pyarrow
from pyarrow import feather


X, Y, Z = 0, 1, 2
XYZUVW = list('xyzuvw')
IJK = list('ijk')
XYZ = list('xyz')


class ErrorCloseToZero(Exception):
    '''Raised if normalizing if sum of probabilities is close to zero'''
    pass


@contextmanager
def ignore_exception(exc):
    '''ignore exception `exc`'''
    try:
        yield
    except exc:
        pass


def write_feather(path, df):
    '''Write a DataFrame to disk using feather serialization format

    Note: This performs destructive changes to the dataframe, caller must
    save it if they need an unchanged version
    '''
    assert path.endswith('.feather'), 'Can only write feathers at the moment'

    df.columns = map(str, df.columns)
    df.reset_index(drop=True, inplace=True)
    feather.write_feather(df, path)


def read_feather(path, columns=None):
    '''Read a feather from disk, with specified columns'''
    # this turns off mmap, and makes the read *much* (>10x) faster on GPFS
    source = pyarrow.OSFile(path)
    return feather.FeatherReader(source).read_pandas(columns)


def load(filename):
    """Load a Pandas/Nrrd file based on the extension"""
    extension = os.path.splitext(filename)[1]
    try:
        return {
            '.feather': lambda: read_feather(filename),
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
    # FLATIndex is not threadsafe, and it leaks memory; to work around that
    # a the process pool forks a new process, and only runs 100 (b/c chunksize=100)
    # iterations before forking a new process (b/c maxtasksperchild=1)
    cpu_count = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(cpu_count)
    ret = pool.map(func, *it, chunksize=50)  # pylint: disable=no-value-for-parameter
    pool.close()
    pool.join()
    del pool
    return ret


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


def mask_by_region_ids(annotation_raw, region_ids):
    '''get a binary voxel mask where the voxel belonging to the given region ids are True'''

    in_region = np.in1d(annotation_raw, list(region_ids))
    in_region = in_region.reshape(np.shape(annotation_raw))
    return in_region


def mask_by_region_names(annotation_raw, hierarchy, names):
    '''get a binary voxel mask where the voxel belonging to the given region names are True'''
    all_ids = []
    for n in names:
        ids = hierarchy.collect('name', n, 'id')
        if not ids:
            raise KeyError(n)
        all_ids.extend(ids)

    return mask_by_region_ids(annotation_raw, all_ids)


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
        mask = mask_by_region_names(atlas.raw, hierarchy, [region])
    else:
        region_ids = list(chain.from_iterable(hierarchy.collect('id', id_, 'id')
                                              for id_ in region))

        mask = mask_by_region_ids(atlas.raw, region_ids)
    return mask
