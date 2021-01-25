'''Utils for projectionizer'''
from contextlib import contextmanager
import json
import logging
import os
from itertools import chain
import multiprocessing
import re

import numpy as np
import pandas as pd
from voxcell import VoxelData
import pyarrow
from pyarrow import feather
import yaml

from bluepy_configfile.configfile import BlueConfig


X, Y, Z = 0, 1, 2
XYZUVW = list('xyzuvw')
IJK = list('ijk')
XYZ = list('xyz')


class ErrorCloseToZero(Exception):
    '''Raised if normalizing if sum of probabilities is close to zero'''


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


def read_yaml(path):
    '''Read a yaml file from given path'''
    with open(path, 'r') as fd:
        return yaml.load(fd, Loader=yaml.Loader)


def load(filename):
    """Load a Pandas/Nrrd file based on the extension"""
    extension = os.path.splitext(filename)[1]
    try:
        return {
            '.feather': lambda: read_feather(filename),
            '.nrrd': lambda: VoxelData.load_nrrd(filename),
            '.csv': lambda: pd.read_csv(filename, index_col=0),
            '.json': lambda: json.load(open(filename)),
            '.yaml': lambda: read_yaml(filename),
        }[extension]()
    except KeyError:
        raise NotImplementedError('Do not know how open: {}'.format(filename))


def load_all(inputs):
    '''load all `inputs`'''
    return [load(x.path) for x in inputs]


def map_parallelize(func, it, jobs=36, chunksize=100):
    '''apply func to all items in it, using a process pool'''
    if os.environ.get('PARALLEL_VERBOSE', False):
        from multiprocessing import util
        util.log_to_stderr(logging.DEBUG)

    jobs = os.environ.get('PARALLEL_COUNT', jobs)

    # FLATIndex is not threadsafe, and it leaks memory; to work around that
    # a the process pool forks a new process, and only runs 100 (b/c chunksize=100)
    # iterations before forking a new process (b/c maxtasksperchild=1)
    pool = multiprocessing.Pool(jobs, maxtasksperchild=1)
    ret = pool.map(func, it, chunksize)  # pylint: disable=no-value-for-parameter
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
    ret = ((min_xyz[X] < df['x'].values) & (df['x'].values < max_xyz[X]) &
           (min_xyz[Y] < df['y'].values) & (df['y'].values < max_xyz[Y]) &
           (min_xyz[Z] < df['z'].values) & (df['z'].values < max_xyz[Z]))
    return pd.Series(ret, index=df.index)


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


def mask_by_region_acronyms(annotation_raw, region_map, acronyms):
    '''get a binary voxel mask where the voxel belonging to the given region acronyms are True'''
    all_ids = []
    for n in acronyms:
        ids = region_map.find(n, 'acronym', with_descendants=True)
        if not ids:
            raise KeyError(n)
        all_ids.extend(ids)

    return mask_by_region_ids(annotation_raw, all_ids)


def mask_by_region(regions, atlas):
    '''
    Args:
        region(str or list of region ids): name/ids to look up in atlas
        path(str): path to where nrrd files are, must include 'brain_regions.nrrd'
    '''
    brain_regions = atlas.load_data('brain_regions')
    region_map = atlas.load_region_map()
    if all([isinstance(reg, int) for reg in regions]):
        region_ids = list(chain.from_iterable(region_map.find(id_, 'id', with_descendants=True)
                                              for id_ in regions))
        mask = mask_by_region_ids(brain_regions.raw, region_ids)
    else:
        mask = mask_by_region_acronyms(brain_regions.raw, region_map, regions)
    return mask


def _regex_to_regions(region_str):
    '''Convert the region regex string in manifest to list of regions'''
    # Replace @, ^, $, (, ), \ with and empty string and split on |
    return re.sub(r'[\@\^\$\(\)\\]', '', region_str).split('|')


def read_regions_from_manifest(circuit_config):
    '''Read the regions from the MANIFEST.yaml'''
    with open(circuit_config, 'r') as fd:
        bc = BlueConfig(fd)

    if hasattr(bc.Run, 'BioName'):
        manifest = load(os.path.join(bc.Run.BioName, 'MANIFEST.yaml'))

        if ('common' in manifest) and ('region' in manifest['common']):
            return _regex_to_regions(manifest['common']['region'])

    return []
