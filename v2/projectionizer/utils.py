'''Utils for projectionizer'''
import json
import os
import re
from itertools import chain
from multiprocessing import Pool
from types import StringTypes

import numpy as np
import pandas as pd
from luigi import Config, FloatParameter, IntParameter, Parameter
from luigi.local_target import LocalTarget
from voxcell import Hierarchy, VoxelData, build

IJK = list('ijk')
X, Y, Z = 0, 1, 2


class ErrorCloseToZero(Exception):
    '''Raised if normalizing if sum of probabilities is close to zero'''


def _write_feather(name, df):
    """Write a DataFrame to disk using feather serialization format

    Note: This performs destructive changes to the dataframe, caller must
    save it if they need an unchanged version
    """
    df.columns = map(str, df.columns)
    df = df.reset_index(drop=True)
    df.to_feather(name)


def _camel_case_to_spinal_case(name):
    '''Camel case to snake case'''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()


class CommonParams(Config):
    """Paramaters that must be passed to all Task"""
    circuit_config = Parameter()
    folder = Parameter()
    geometry = Parameter()
    n_total_chunks = IntParameter()
    sgid_offset = IntParameter()
    oversampling = FloatParameter()

    # S1HL/S1 region parameters
    voxel_path = Parameter(default='j')
    prefix = Parameter(default='')

    extension = None

    def output(self):

        name = _camel_case_to_spinal_case(self.__class__.__name__)
        if hasattr(self, 'chunk_num'):
            return LocalTarget('{}/{}-{}.{}'.format(self.folder,
                                                    name,
                                                    getattr(self, 'chunk_num'),
                                                    self.extension))
        return LocalTarget('{}/{}.{}'.format(self.folder, name, self.extension))


class FeatherTask(CommonParams):
    '''Task returning a feather file'''
    extension = 'feather'


class JsonTask(CommonParams):
    '''Task returning a JSON file'''
    extension = 'json'


class NrrdTask(CommonParams):
    '''Task returning a Nrrd file'''
    extension = 'nrrd'


def load(filename):
    """Load a Pandas/Nrrd file based on the extension"""
    if filename.endswith('feather'):
        return pd.read_feather(filename)
    elif filename.endswith('nrrd'):
        return VoxelData.load_nrrd(filename)
    elif filename.endswith('json'):
        with open(filename) as infile:
            return json.load(infile)
    raise NotImplementedError('Do not know how open: {}'.format(filename))


def load_all(inputs):
    '''load all `inputs`'''
    return [load(x.path) for x in inputs]


def cloned_tasks(this, tasks):
    '''Utils function for self.requires()
    Returns: clone and returns a list of required tasks'''
    return [this.clone(task) for task in tasks]


def map_parallelize(func, *it):
    '''apply func to all items in it, using a process pool

    Watch the memory usage!
    '''
    pool = Pool(14)
    ret = pool.map(func, *it)  # pylint: disable=no-value-for-parameter
    pool.close()
    pool.join()
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
    if isinstance(region, StringTypes):
        mask = build.mask_by_region_names(atlas.raw, hierarchy, [region])
    else:
        region_ids = list(chain.from_iterable(hierarchy.collect('id', id_, 'id')
                                              for id_ in region))

        mask = build.mask_by_region_ids(atlas.raw, region_ids)
    return mask
