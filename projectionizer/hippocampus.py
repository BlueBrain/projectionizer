"""sampling utils"""

import logging
from functools import partial

import numpy as np
import pandas as pd
import spatial_index
import spatial_index.experimental

from projectionizer import synapses, utils
from projectionizer.utils import write_feather

L = logging.getLogger(__name__)

SEGMENT_COLUMNS = (
    [
        "section_id",
        "segment_id",
        "segment_length",
        "section_type",
    ]
    + synapses.SEGMENT_START_COLS
    + synapses.SEGMENT_END_COLS
    + ["gid"]
)


def _full_sample_worker(min_xyzs, index_path, voxel_dimensions):
    """for every voxel defined by the lower coordinate in min_xyzs, gather segments

    Args:
        min_xyzs(np.array of (Nx3): lower coordinates of voxels
        index_path(Path): absolute path spatial index `MultiIndex`
        voxel_dimensions(1x3 array): voxel dimensions
    """
    dfs = []
    index = spatial_index.open_index(str(index_path))
    for min_xyz in min_xyzs:
        segs_df = synapses.pick_segments_voxel(
            index,
            min_xyz,
            min_xyz + voxel_dimensions,
            dataframe_cleanup=synapses.downcast_int_columns,
            drop_axons=True,
        )
        if segs_df is not None:
            dfs.append(segs_df[SEGMENT_COLUMNS])

    if len(dfs):
        df = pd.concat(dfs, ignore_index=True, sort=False)
    else:
        df = pd.DataFrame(columns=SEGMENT_COLUMNS)

    return df


def full_sample_parallel(brain_regions, region, region_id, index_path, output):
    """Sample *all* segments of type region_id

    Args:
        brain_regions(VoxelData): brain regions
        region(str): name of the region to sample
        region_id(int): single region id to sample
        index_path(Path): absolute path spatial index `MultiIndex`
        output(Path): directory where to save the data
    """
    nz = np.array(np.nonzero(brain_regions.raw == region_id)).T
    if len(nz) == 0:
        return

    positions = brain_regions.indices_to_positions(nz)
    positions = np.unique(positions, axis=0)
    order = spatial_index.experimental.space_filling_order(positions)
    positions = positions[order]

    chunks = (len(positions) // 500000) + 1

    func = partial(
        _full_sample_worker, index_path=index_path, voxel_dimensions=brain_regions.voxel_dimensions
    )
    for i, xyzs in enumerate(np.array_split(positions, chunks, axis=0)):
        path = output / f"{region}_{region_id}_{i:03d}.feather"

        if path.exists():
            L.info("Already did: %s", path)
            continue

        with utils.delete_file_on_exception(path):
            df = utils.map_parallelize(func, np.array_split(xyzs, (len(xyzs) // 10000) + 1, axis=0))
            df = pd.concat(df, ignore_index=True, sort=False)
            df.rename(columns={"gid": "tgid"}, inplace=True)

            write_feather(path, df)
