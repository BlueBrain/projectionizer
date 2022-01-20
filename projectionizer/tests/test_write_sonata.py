import os

import h5py
import pandas as pd
from numpy.testing import assert_array_equal, assert_equal

from projectionizer import write_sonata

from utils import EDGE_POPULATION, NODE_POPULATION, setup_tempdir


def test_write_nodes():
    df = pd.DataFrame({'tgid': [10, 15],
                       'sgid': [20, 13],
                       'x': [0.45] * 2,
                       'y': [0.33] * 2,
                       'z': [0.11] * 2,
                       'section_id': [1033] * 2,
                       'synapse_offset': [128.] * 2,
                       'afferent_indices': [12] * 2,
                       'segment_id': [1033] * 2,
                       'section_type': [3] * 2,
                       'sgid_path_distance': [300.] * 2
                       })

    with setup_tempdir('test_utils') as folder:
        path = os.path.join(folder, 'sscx_nodes.sonata')
        mtype = 'fake_type'
        write_sonata.write_nodes(df, path, NODE_POPULATION, mtype, keep_offset=True)

        with h5py.File(path, 'r') as f:
            # size should be df.sgid.max() since keep_offset=True
            sscx_proj = f[f'nodes/{NODE_POPULATION}']
            assert_equal(sscx_proj['node_type_id'].size, df.sgid.max())
            assert_array_equal(sscx_proj['node_type_id'][:], [-1] * df.sgid.max())
            keys = sscx_proj['0']['@library'].keys()

            for k in keys:
                assert_equal(sscx_proj['0'][k].size, sscx_proj['node_type_id'].size)
                assert_array_equal(sscx_proj['0'][k][:], [0] * df.sgid.max())

    with setup_tempdir('test_utils') as folder:
        path = os.path.join(folder, 'sscx_nodes.sonata')
        mtype = 'fake_type'
        write_sonata.write_nodes(df, path, NODE_POPULATION, mtype, keep_offset=False)

        with h5py.File(path, 'r') as f:
            sscx_proj = f[f'nodes/{NODE_POPULATION}']
            # size should be df.sgid.max() - df.sgid.min() + 1 since keep_offset=False
            assert_equal(sscx_proj['node_type_id'].size, df.sgid.max() - df.sgid.min() + 1)


def test_write_edges():
    df = pd.DataFrame({'tgid': [10, 15],
                       'sgid': [20, 13],
                       'x': [0.45] * 2,
                       'y': [0.33] * 2,
                       'z': [0.11] * 2,
                       'section_id': [1033] * 2,
                       'synapse_offset': [128.] * 2,
                       'afferent_indices': [12] * 2,
                       'segment_id': [1033] * 2,
                       'section_type': [3] * 2,
                       'sgid_path_distance': [300.] * 2
                       })

    with setup_tempdir('test_utils') as folder:
        path = os.path.join(folder, 'sscx_edges.sonata')
        write_sonata.write_edges(df, path, EDGE_POPULATION, keep_offset=True)

        with h5py.File(path, 'r') as f:
            sscx_proj = f[f'edges/{EDGE_POPULATION}']
            assert_equal(sscx_proj['edge_type_id'].size, len(df.sgid))
            assert_array_equal(sscx_proj['edge_type_id'][:], [-1] * len(df.sgid))
            # should return data.sgid - 1 since keep_offset=True and df is converted to 0-indexed
            assert_array_equal(sscx_proj['source_node_id'][:], df.sgid - 1)
            # should return data.tgid - 1 since df is converted to 0-indexed
            assert_array_equal(sscx_proj['target_node_id'][:], df.tgid - 1)

            attributes = sscx_proj['0']
            assert_array_equal(attributes['afferent_center_x'], df.x)
            assert_array_equal(attributes['afferent_center_y'], df.y)
            assert_array_equal(attributes['afferent_center_z'], df.z)
            assert_array_equal(attributes['distance_soma'], df.sgid_path_distance)
            assert_array_equal(attributes['afferent_section_id'], df.section_id)
            assert_array_equal(attributes['afferent_segment_id'], df.segment_id)
            assert_array_equal(attributes['afferent_segment_offset'], df.synapse_offset)
            assert_array_equal(attributes['afferent_section_type'], df.section_type)
            assert_array_equal(attributes['efferent_section_type'], [2] * 2)

    with setup_tempdir('test_utils') as folder:
        path = os.path.join(folder, 'sscx_edges.sonata')
        write_sonata.write_edges(df, path, EDGE_POPULATION, keep_offset=False)

        with h5py.File(path, 'r') as f:
            sscx_proj = f[f'edges/{EDGE_POPULATION}']
            assert_equal(sscx_proj['edge_type_id'].size, 2)
            # should return df.sgid-df.sgid.min() (smallest sgid = 0) since keep_offset=False
            assert_array_equal(sscx_proj['source_node_id'][:], df.sgid - df.sgid.min())
            # should still return data.tgid - 1
            assert_array_equal(sscx_proj['target_node_id'][:], df.tgid - 1)
