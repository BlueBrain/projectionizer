'''Utils for projectionizer'''
import json
from multiprocessing import Pool
from os.path import join

from types import StringTypes

import luigi
import numpy as np
import pandas as pd
import voxcell
from dask.distributed import Client


IJK = list('ijk')
X, Y, Z = 0, 1, 2


class ErrorCloseToZero(Exception):
    '''Raised if normalizing if sum of probabilities is close to zero'''
    pass


def _write_feather(name, df):
    """Write a DataFrame to disk using feather serialization format

    Note: This performs destructive changes to the dataframe, caller must
    save it if they need an unchanged version
    """
    df.columns = map(str, df.columns)
    df = df.reset_index(drop=True)
    df.to_feather(name)


class CommonParams(luigi.Config):
    """Paramaters that must be passed to all Task"""
    circuit_config = luigi.Parameter()
    folder = luigi.Parameter()
    geometry = luigi.Parameter()
    n_total_chunks = luigi.IntParameter()
    sgid_offset = luigi.IntParameter()

    # S1HL/S1 region parameters
    voxel_path = luigi.Parameter(default=None)
    prefix = luigi.Parameter(default=None)


def load(filename):
    """Load a Pandas/Nrrd file based on the extension"""
    if filename.endswith('feather'):
        return pd.read_feather(filename)
    elif filename.endswith('nrrd'):
        return voxcell.VoxelData.load_nrrd(filename)
    raise Exception('Do not know how open: {}'.format(filename))


def load_all(inputs):
    return [load(x.path) for x in inputs]


def cloned_tasks(this, tasks):
    '''Utils function for self.requires()
    Returns: clone and returns a list of required tasks'''
    return [this.clone(task) for task in tasks]


def map_parallelize(func, *it):
    pool = Pool(14)
    ret = pool.map(func, *it)
    pool.close()
    pool.join()
    return ret


def normalize_probability(p, axis=None):
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
    atlas = voxcell.VoxelData.load_nrrd(join(path, prefix + 'brain_regions.nrrd'))
    with open(join(path, 'hierarchy.json')) as fd:
        hierarchy = voxcell.Hierarchy(json.load(fd))
    if isinstance(region, StringTypes):
        mask = voxcell.build.mask_by_region_names(atlas.raw, hierarchy, [region])
    else:
        region_ids = []
        for id_ in region:
            region_ids.extend(hierarchy.collect('id', id_, 'id'))

        mask = voxcell.build.mask_by_region_ids(atlas.raw, region_ids)
    return mask
