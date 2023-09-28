import h5py
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_equal

import projectionizer.write_sonata as test_module

from utils import EDGE_POPULATION, NODE_POPULATION


def test_write_nodes(tmp_confdir):
    df = pd.DataFrame(
        {
            "tgid": [10, 15],
            "sgid": [20, 13],
            "x": [0.45] * 2,
            "y": [0.33] * 2,
            "z": [0.11] * 2,
            "section_id": [1033] * 2,
            "section_pos": [0.5] * 2,
            "synapse_offset": [128.0] * 2,
            "afferent_indices": [12] * 2,
            "segment_id": [1033] * 2,
            "section_type": [3] * 2,
            "sgid_path_distance": [300.0] * 2,
        }
    )

    path = tmp_confdir / "sscx_nodes.sonata"
    mtype = "fake_type"
    test_module.write_nodes(df, path, NODE_POPULATION, mtype, keep_offset=True)

    with h5py.File(path, "r") as f:
        # size should be df.sgid.max() + 1 since keep_offset=True and 0-based index
        expected_size = df.sgid.max() + 1
        sscx_proj = f[f"nodes/{NODE_POPULATION}"]
        assert_equal(sscx_proj["node_type_id"].size, expected_size)
        assert_array_equal(sscx_proj["node_type_id"][:], [-1] * expected_size)
        keys = sscx_proj["0"]["@library"].keys()

        for k in keys:
            assert_equal(sscx_proj["0"][k].size, sscx_proj["node_type_id"].size)
            assert_array_equal(sscx_proj["0"][k][:], [0] * expected_size)

    path = tmp_confdir / "sscx_nodes_no_offset.sonata"
    mtype = "fake_type"
    test_module.write_nodes(df, path, NODE_POPULATION, mtype, keep_offset=False)

    with h5py.File(path, "r") as f:
        sscx_proj = f[f"nodes/{NODE_POPULATION}"]
        # size should be df.sgid.max() - df.sgid.min() + 1 since keep_offset=False
        assert_equal(sscx_proj["node_type_id"].size, df.sgid.max() - df.sgid.min() + 1)


def test_write_edges(tmp_confdir):
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "tgid": [10, 15],
            "sgid": [20, 13],
            "x": np.random.random(2),
            "y": np.random.random(2),
            "z": np.random.random(2),
            "source_x": np.random.random(2),
            "source_y": np.random.random(2),
            "source_z": np.random.random(2),
            "section_id": [1033] * 2,
            "section_pos": [0.5] * 2,
            "synapse_offset": [128.0] * 2,
            "afferent_indices": [12] * 2,
            "segment_id": [1033] * 2,
            "section_type": [3] * 2,
            "sgid_path_distance": [300.0] * 2,
            "distance_volume_transmission": np.random.random(2),
        }
    )

    path = tmp_confdir / "sscx_edges.sonata"
    test_module.write_edges(df, path, EDGE_POPULATION, keep_offset=True)

    with h5py.File(path, "r") as f:
        sscx_proj = f[f"edges/{EDGE_POPULATION}"]
        assert_equal(sscx_proj["edge_type_id"].size, len(df.sgid))
        assert_array_equal(sscx_proj["edge_type_id"][:], [-1] * len(df.sgid))
        # should return df.sgid since keep_offset=True
        assert_array_equal(sscx_proj["source_node_id"][:], df.sgid)
        assert_array_equal(sscx_proj["target_node_id"][:], df.tgid)

        attributes = sscx_proj["0"]
        assert_array_equal(attributes["afferent_center_x"], df.x)
        assert_array_equal(attributes["afferent_center_y"], df.y)
        assert_array_equal(attributes["afferent_center_z"], df.z)
        assert_array_equal(attributes["efferent_center_x"], df.source_x)
        assert_array_equal(attributes["efferent_center_y"], df.source_y)
        assert_array_equal(attributes["efferent_center_z"], df.source_z)
        assert_array_equal(attributes["distance_soma"], df.sgid_path_distance)
        assert_array_equal(attributes["afferent_section_id"], df.section_id)
        assert_array_equal(attributes["afferent_section_pos"], df.section_pos)
        assert_array_equal(attributes["afferent_segment_id"], df.segment_id)
        assert_array_equal(attributes["afferent_segment_offset"], df.synapse_offset)
        assert_array_equal(attributes["afferent_section_type"], df.section_type)
        assert_array_equal(attributes["efferent_section_type"], [2] * 2)
        assert_array_equal(
            attributes["distance_volume_transmission"], df.distance_volume_transmission
        )

    path = tmp_confdir / "sscx_edges_no_offset.sonata"
    test_module.write_edges(df, path, EDGE_POPULATION, keep_offset=False)

    with h5py.File(path, "r") as f:
        sscx_proj = f[f"edges/{EDGE_POPULATION}"]
        assert_equal(sscx_proj["edge_type_id"].size, 2)
        # should return df.sgid-df.sgid.min() (smallest sgid = 0) since keep_offset=False
        assert_array_equal(sscx_proj["source_node_id"][:], df.sgid - df.sgid.min())
        assert_array_equal(sscx_proj["target_node_id"][:], df.tgid)
