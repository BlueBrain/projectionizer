import logging
from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd
from morphio import Morphology
from numpy.testing import assert_array_almost_equal, assert_array_equal

import projectionizer.afferent_section_position as test_module

from utils import TEST_DATA_DIR, setup_tempdir


def create_dummy_node_file(dirpath):
    node_path = Path(dirpath, "nodes.h5")
    with h5py.File(node_path, "w") as h5:
        pop = h5.create_group("/nodes/dummy/")
        pop["node_type_id"] = [-1]
        pop["0/morphology"] = ["morph"]

    return node_path


def test_load_morphology():
    expected = Morphology(Path(TEST_DATA_DIR, "morph.swc"))
    morph = test_module.load_morphology(
        morph_name="morph", morph_path=TEST_DATA_DIR, morph_type="swc"
    )

    for m, e in zip(morph.sections, expected.sections):
        assert_array_equal(m.points, e.points)


def test_compute_afferent_section_pos(caplog):
    morph = Morphology(Path(TEST_DATA_DIR, "morph.swc"))
    Row = namedtuple("Row", ["section_id", "segment_id", "synapse_offset", "orig_index"])

    # Test for the beginning of the section
    res = test_module.compute_afferent_section_pos(Row(1, 0, 0, 0), morph)
    assert_array_equal(res, 0)

    # Test for the end of the section
    last_seg_id = morph.section(0).n_points - 2
    last_seg_len = np.linalg.norm(np.diff(morph.section(0).points[-2:], axis=0))
    with caplog.at_level(logging.WARNING):
        res = test_module.compute_afferent_section_pos(Row(1, last_seg_id, last_seg_len, 0), morph)
    assert "Value exceeds threshold" not in caplog.text
    assert_array_equal(res, 1)

    # Test for going over the last section more than WARNING_THRESHOLD.
    # This should still return 1 as afferent_section_pos and print a warning.
    last_seg_len += test_module.WARNING_THRESHOLD + 1e6
    with caplog.at_level(logging.WARNING):
        res = test_module.compute_afferent_section_pos(Row(1, last_seg_id, last_seg_len, 0), morph)
    assert "Value exceeds threshold" in caplog.text
    assert_array_equal(res, 1)

    # Test a few manually calculated points
    assert_array_almost_equal(
        test_module.compute_afferent_section_pos(Row(1, 0, 3.0, 0), morph), 0.2317627
    )
    assert_array_almost_equal(
        test_module.compute_afferent_section_pos(Row(2, 2, 1.0, 0), morph), 0.8333333
    )

    # For future-proofing, select randomly some samples
    np.random.seed(42)
    res = []
    section_ids = np.random.randint(1, len(morph.sections) + 1, 10)
    for section_id in section_ids:
        segment_id = np.random.randint(0, morph.section(section_id - 1).n_points - 1)
        segment_limits = morph.section(section_id - 1).points[segment_id : segment_id + 2]
        segment_len = np.linalg.norm(np.diff(segment_limits, axis=0), axis=1)[0]
        offset = np.random.random() * segment_len
        row = Row(section_id, segment_id, offset, 0)
        res.append(test_module.compute_afferent_section_pos(row, morph))

    expected = [
        0.15599452,
        0.05808361,
        0.89705235,
        0.14286682,
        0.6508885,
        0.05641158,
        0.97411096,
        0.34576055,
        0.9922116,
        0.10141408,
    ]

    assert_array_almost_equal(res, expected)


@patch(f"{test_module.__name__}.load_morphology", Mock())
def test_compute_positions_worker():
    df = pd.DataFrame({"orig_index": np.arange(10, 20)})

    with patch(f"{test_module.__name__}.compute_afferent_section_pos") as patched:
        patched.return_value = 1
        res = test_module.compute_positions_worker(
            morph_df=("morph", df),
            morph_path="fake/path",
            morph_type="fake",
        )

    assert_array_equal(res[1], df.orig_index)
    assert res[0].dtype == np.dtype(np.float32)


def test_get_morphs_for_nodes():
    with setup_tempdir("get_morphs_for_nodes") as temp_dir:
        node_path = create_dummy_node_file(temp_dir)
        morphs = test_module.get_morphs_for_nodes(node_path, population="dummy")

    pd.testing.assert_frame_equal(pd.DataFrame(["morph"], columns=["morph"], index=[1]), morphs)


def test_compute_positions():
    df = pd.DataFrame(
        {
            "tgid": np.zeros(1),
            "section_id": np.zeros(1),
            "segment_id": np.zeros(1),
            "synapse_offset": np.zeros(1),
        }
    )

    a_1_10 = np.random.permutation(np.arange(10))
    a_10_20 = np.random.permutation(np.arange(10)) + 10

    with patch(f"{test_module.__name__}.get_morphs_for_nodes") as patch_morphs:
        patch_morphs.return_value = pd.DataFrame({"morph": "morph"}, index=[1])

        with patch(f"{test_module.__name__}.map_parallelize") as patch_map:
            # To check that the sorting the result based on the indexes works
            patch_map.return_value = (
                (np.copy(a_1_10), np.copy(a_1_10)),
                (np.copy(a_10_20), np.copy(a_10_20)),
            )

            res = test_module.compute_positions(
                synapses=df,
                node_path="fake/path",
                node_population="fake",
                morph_path="fake/path",
                morph_type="fake",
            )

    assert_array_equal(res.section_pos, np.arange(20))
    assert np.dtype(res.section_pos) == np.dtype(np.float32)
