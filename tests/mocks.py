import numpy as np
import pandas as pd
from voxcell import VoxelData


def create_synapse_counts():
    synapse_counts = VoxelData(np.zeros((5, 5, 5)), [10] * 3, (10, 10, 10))
    idx = np.array(
        [
            [2, 2, 2],  # middle of cube: xyz = (35, 35, 35)
            [2, 2, 3],  # xyz = (35, 35, 45)
        ]
    )
    synapse_counts.raw[tuple(idx.T)] = 1
    return synapse_counts


def create_virtual_fibers():
    data = np.array(
        [
            [30.0, 30.0, 30.0, 1.0, 0.0, 0.0],  # (30, 30, 30), along x axis
            [30.0, 30.0, 30.0, 0.0, 1.0, 0.0],  # (30, 30, 30), along y axis
            [30.0, 30.0, 30.0, 0.0, 0.0, 1.0],  # (30, 30, 30), along z axis
        ]
    )
    virtual_fibers = pd.DataFrame(data, columns=list("xyzuvw"))
    return virtual_fibers


def create_candidates():
    return pd.DataFrame(
        [
            [1, 2, 0, 30.5, 30.5, 30.5],
            [1, 2, 0, 30.5, 30.5, 30.5],
            [1, 2, 0, 30.5, 30.5, 30.5],
        ],
        columns=["0", "1", "2", "x", "y", "z"],
    )
