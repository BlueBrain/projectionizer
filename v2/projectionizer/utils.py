import numpy as np

from bluepy.v2.enums import Cell
from bluepy.v2.enums import Cell, Section
from neurom import NeuriteType


class ErrorCloseToZero(Exception):
    pass


IJK = list('ijk')
XYZ = [Cell.X, Cell.Y, Cell.Z]


def normalize_probability(p):
    """ Normalize vector of probabilities `p` so that sum(p) == 1. """
    norm = np.sum(p)
    if norm < 1e-7:
        raise ErrorCloseToZero("Could not normalize almost-zero vector")
    return p / norm


def voxelize_cells(cells, voxel_size):
    cells[IJK] = (cells[XYZ] // voxel_size).astype(np.int)


def segment_pref(segs_df):
    '''don't want axons, assign probability of 0 to them, and 1 to other neurite types,
    this will be normalized by the caller
    '''
    return (segs_df[Section.NEURITE_TYPE] != NeuriteType.axon).astype(float)
