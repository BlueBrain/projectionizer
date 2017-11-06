'''Utils for projectionizer'''
import numpy as np

from bluepy.v2.enums import Section, Segment
from neurom import NeuriteType
from dask.distributed import Client


class ErrorCloseToZero(Exception):
    '''Raised if normalizing if sum of probabilities is close to zero'''
    pass


# bluepy.v2 returns a DataFrame with the start and endpoint of the segments when performing a query,
# simplify addressing them using the following
SEGMENT_START_COLS = [Segment.X1, Segment.Y1, Segment.Z1, ]
SEGMENT_END_COLS = [Segment.X2, Segment.Y2, Segment.Z2, ]

IJK = list('ijk')
X, Y, Z = 0, 1, 2


def normalize_probability(p):
    """ Normalize vector of probabilities `p` so that sum(p) == 1. """
    norm = np.sum(p)
    if norm < 1e-7:
        raise ErrorCloseToZero("Could not normalize almost-zero vector")
    return p / norm


def segment_pref(df):
    '''don't want axons, assign probability of 0 to them, and 1 to other neurite types,
    this will be normalized by the caller
    '''
    return (df[Section.NEURITE_TYPE] != NeuriteType.axon).astype(float)

def segment_pref_length(df):
    '''don't want axons, assign probability of 0 to them, and 1 to other neurite types,
    multiplied by the length of the segment
    this will be normalized by the caller
    '''
    return df['segment_length'] * (df[Section.NEURITE_TYPE] != NeuriteType.axon).astype(float)


def in_bounding_box(min_xyz, max_xyz, df):
    '''return boolean index of df rows that are in min_xyz/max_xyz

    df must have ['x', 'y', 'z'] columns
    '''
    ret = ((min_xyz[X] < df['x']) & (df['x'] < max_xyz[X]) &
           (min_xyz[Y] < df['y']) & (df['y'] < max_xyz[Y]) &
           (min_xyz[Z] < df['z']) & (df['z'] < max_xyz[Z]))
    return ret


def map_func(parallelize):
    if parallelize:
        client = Client(memory_limit=5e9)

        def map_(func, it):
            res = client.map(func, it)
            return client.gather(res)
        return map_
    else:
        return map
