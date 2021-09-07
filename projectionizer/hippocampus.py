'''sampling utils'''
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
from neurom import NeuriteType

from projectionizer import synapses, utils
from projectionizer.utils import (
    convert_to_smallest_allowed_int_type,
    in_bounding_box,
    write_feather,
)

L = logging.getLogger(__name__)

SAMPLE_PATH = 'SAMPLED'

SEGMENT_START_COLS = ['segment_x1', 'segment_y1', 'segment_z1', ]
SEGMENT_END_COLS = ['segment_x2', 'segment_y2', 'segment_z2', ]
SEGMENT_COLUMNS = sorted(['section_id', 'segment_id', 'segment_length', ] +
                         SEGMENT_START_COLS + SEGMENT_END_COLS +
                         ['tgid']
                         )


def _min_max_axis(min_xyz, max_xyz):
    '''get min/max axis'''
    return np.minimum(min_xyz, max_xyz), np.maximum(min_xyz, max_xyz)


def _full_sample_worker(min_xyzs, index_path, voxel_dimensions):
    '''for every voxel defined by the lower coordinate in min_xzys, gather segments

    Args:
        min_xyzs(np.array of (Nx3):
        index_path(str): path to FlatIndex indices
        voxel_dimensions(1x3 array): voxel dimensions
    '''
    start_cols = ['Segment.X1', 'Segment.Y1', 'Segment.Z1']
    end_cols = ['Segment.X2', 'Segment.Y2', 'Segment.Z2', ]

    chunks = []
    for min_xyz in min_xyzs:
        max_xyz = min_xyz + voxel_dimensions
        df = synapses._sample_with_flat_index(  # pylint: disable=protected-access
            index_path, min_xyz, max_xyz)

        if df is None or len(df) == 0:
            continue

        df.columns = map(str, df.columns)

        df = df[df['Section.NEURITE_TYPE'] != NeuriteType.axon] \
            .copy()  # pylint:disable=unsubscriptable-object

        if df is None or len(df) == 0:
            continue

        starts, ends = df[start_cols].values, df[end_cols].values

        # keep only the segments whose midpoints are in the current voxel
        locations = pd.DataFrame((ends + starts) / 2., columns=list('xyz'), index=df.index)
        df = df[in_bounding_box(*_min_max_axis(min_xyz, max_xyz), df=locations)]

        if df is None or len(df) == 0:
            continue

        starts, ends = df[start_cols].values, df[end_cols].values
        df['segment_length'] = np.linalg.norm(ends - starts, axis=1).astype(np.float32)

        #  { need to get rid of memory usage as quickly as possible
        #    MOs5 (the largest region by voxel count) *barely* fits into 300GB
        def fix_name(name):
            '''convert pandas column names to snake case'''
            return name.lower().replace('.', '_')

        # float64 -> float32
        for name in start_cols + end_cols:
            df[fix_name(name)] = df[name].values.astype(np.float32)
            del df[name]

        # uint -> smallest uint needed
        for name in ('Section.ID', 'Segment.ID', ):
            df[fix_name(name)] = convert_to_smallest_allowed_int_type(df[name])
            del df[name]

        df['tgid'] = convert_to_smallest_allowed_int_type(df['gid'])

        del df['Section.NEURITE_TYPE'], df['Segment.R1'], df['Segment.R2'], df['gid']
        #  }

        chunks.append(df)

    if len(chunks):
        df = pd.concat(chunks, ignore_index=True, sort=False)
    else:
        df = pd.DataFrame(columns=SEGMENT_COLUMNS)
    return df


def full_sample_parallel(brain_regions, region, region_id, index_path, output):
    '''Sample *all* segments of type region_id

    Args:
        brain_regions(VoxelData): brain regions
        region_id(int): single region id to sample
        index_path(str): directory where FLATIndex can find SEGMENT_*
    '''
    output = os.path.join(output, SAMPLE_PATH)
    if not os.path.exists(output):
        os.makedirs(output)

    nz = np.array(np.nonzero(brain_regions.raw == region_id)).T
    if len(nz) == 0:
        return None

    positions = brain_regions.indices_to_positions(nz)
    positions = np.unique(positions, axis=0)
    chunks = (len(positions) // 500000) + 1

    func = partial(_full_sample_worker,
                   index_path=index_path,
                   voxel_dimensions=brain_regions.voxel_dimensions)
    for i, xyzs in enumerate(np.array_split(positions, chunks, axis=0)):
        path = os.path.join(output, '%s_%d_%03d.feather' % (region, region_id, i))

        if os.path.exists(path):
            L.info('Already did: %s', path)
            continue

        with utils.delete_file_on_exception(path):
            df = utils.map_parallelize(func,
                                       np.array_split(xyzs, (len(xyzs) // 10000) + 1, axis=0),
                                       chunksize=1)

            df = pd.concat(df, ignore_index=True, sort=False)
            write_feather(path, df)

    return None
