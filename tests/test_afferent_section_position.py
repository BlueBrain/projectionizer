import logging
from collections import namedtuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from morphio import Morphology
from numpy.testing import assert_array_almost_equal, assert_array_equal

import projectionizer.afferent_section_position as test_module

from utils import TEST_DATA_DIR


def test_compute_afferent_section_pos(caplog):
    morph = Morphology(TEST_DATA_DIR / "morph.swc")
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


@patch.object(test_module, "Morphology", new=Mock())
@patch.object(test_module, "compute_afferent_section_pos", new=Mock(return_value=1))
def test_compute_positions_worker():
    df = pd.DataFrame({"orig_index": np.arange(10, 20)})
    res = test_module.compute_positions_worker(morph_df=("fake/morph/path", df))

    assert_array_equal(res[1], df.orig_index)
    assert res[0].dtype == np.dtype(np.float32)


@patch.object(test_module, "map_parallelize")
def test_compute_positions(mock_map_parallelize):
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

    morphs = pd.DataFrame({"morph": "fake/morph/path"}, index=[1])

    # To check that the sorting the result based on the indexes works
    mock_map_parallelize.return_value = (
        (np.copy(a_1_10), np.copy(a_1_10)),
        (np.copy(a_10_20), np.copy(a_10_20)),
    )

    res = test_module.compute_positions(synapses=df, morphs=morphs)

    assert_array_equal(res.section_pos, np.arange(20))
    assert_array_equal(res.index, np.arange(20))
    assert np.dtype(res.section_pos) == np.dtype(np.float32)
