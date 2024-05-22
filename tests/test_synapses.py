import itertools
import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from morphio import SectionType
from numpy.testing import (
    assert_approx_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from voxcell import VoxelData

import projectionizer.synapses as test_module

from utils import fake_segments


def test_spatial_index_cache_size_env():
    assert isinstance(test_module.CACHE_SIZE_MB, int)
    assert test_module.CACHE_SIZE_MB == 666


def test_segment_pref_length():
    df = pd.DataFrame(
        {
            "section_type": [
                SectionType.axon,
                SectionType.axon,
                SectionType.basal_dendrite,
                SectionType.apical_dendrite,
            ],
            "segment_length": 1,
        }
    )
    ret = test_module.segment_pref_length(df)
    assert isinstance(ret, pd.Series)
    assert_array_equal(ret.values, [0, 0, 1, 1])


def test_get_segment_limits_within_sphere():
    a = np.array(
        [
            [0, 10, -10],
            [0, 6, 6],
            [0, 1, 1],
            [0, -10, -10],
            [-2, -2, -2],
        ],
        dtype=float,
    )
    b = np.array(
        [
            [0, 10, 10],
            [0, 10, 10],
            [0, 10, 10],
            [0, 10, 10],
            [2, 2, 2],
        ],
        dtype=float,
    )
    point = np.array((0, 0, 0))

    res_start, res_end = test_module.get_segment_limits_within_sphere(a, b, point, radius=5)

    expected_start = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [0, 1, 1],
            [0, -np.sqrt(25 / 2), -np.sqrt(25 / 2)],
            [-2, -2, -2],
        ]
    )

    expected_end = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [0, np.sqrt(25 / 2), np.sqrt(25 / 2)],
            [0, np.sqrt(25 / 2), np.sqrt(25 / 2)],
            [2, 2, 2],
        ]
    )

    assert_array_almost_equal(res_start, expected_start)
    assert_array_almost_equal(res_end, expected_end)

    for _ in range(5):
        addition = np.random.random((1, 3)) * np.random.randint(10, 100, (1, 3))
        non_zero_start, non_zero_end = test_module.get_segment_limits_within_sphere(
            a + addition,
            b + addition,
            point + addition,
            radius=5,
        )
        assert_array_almost_equal(non_zero_start, expected_start + addition)
        assert_array_almost_equal(non_zero_end, expected_end + addition)

    # Test that results are same if ends and starts are reversed
    rev_start, rev_end = test_module.get_segment_limits_within_sphere(b, a, point, radius=5)
    assert_array_almost_equal(rev_start, expected_end)
    assert_array_almost_equal(rev_end, expected_start)


@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module, "get_segment_limits_within_sphere")
def test_spherical_sampling(mock_get_limits, mock_sample):
    min_xyz = np.array([10, 10, 10])
    segments = fake_segments(min_xyz, min_xyz + 1, 5)

    mock_sample.return_value = segments

    mock_get_limits.return_value = np.full((2, 5, 3), np.nan)
    mock_get_limits.return_value[0, :2, :] = 1
    mock_get_limits.return_value[1, :2, :] = 2

    res = test_module.spherical_sampling((np.array([0, 0, 0]), 1), "fake_path", radius=5)

    assert len(res) == 2
    assert_array_equal(res.gid, segments.iloc[:2].gid)
    assert {*res.columns} == {*test_module.VOLUME_TRANSMISSION_COLS}


@patch.object(test_module, "_sample_with_spatial_index")
def test_spherical_sampling_prune_zero_length_segments(mock_sample):
    # Test pruning of zero length segments
    min_xyz = np.array([10, 10, 10])
    segments = fake_segments(min_xyz, min_xyz + 1, 5)

    start_pos = test_module.SEGMENT_START_COLS
    end_pos = test_module.SEGMENT_END_COLS

    # Segment starts == Segment ends except for one entry:
    segments[end_pos] = np.full((np.shape(segments)[0], 3), 1)
    segments[start_pos] = segments[end_pos]
    segments.loc[segments.index[1], "segment_x1"] += 1
    segments.loc[segments.index[1], "gid"] = 666

    mock_sample.return_value = segments

    res = test_module.spherical_sampling((np.array([0, 0, 0]), 1), "fake_path", radius=5)
    assert_array_equal(res.gid, segments.iloc[1].gid)


@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test_pick_segments_voxel(mock_sample):
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = "foo/bar/baz"

    # return None if Spatial Index finds nothing
    mock_sample.return_value = pd.DataFrame([])
    segs_df = test_module.pick_segments_voxel(circuit_path, min_xyz, max_xyz)
    assert segs_df is None

    # all segments are axons
    count = 10
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])

    segments = fake_segments(min_xyz, max_xyz, 2 * count)
    segments["section_type"] = SectionType.axon
    mock_sample.return_value = segments
    segs_df = test_module.pick_segments_voxel(circuit_path, min_xyz, max_xyz)
    assert len(segs_df) == 2 * count
    segs_df = test_module.pick_segments_voxel(circuit_path, min_xyz, max_xyz, drop_axons=True)
    assert segs_df is None

    # No segment with midpoint in voxel
    segments = fake_segments(min_xyz, max_xyz, 2 * count)
    mock_sample.return_value = segments
    segs_df = test_module.pick_segments_voxel(circuit_path, 10 + min_xyz, 10 + max_xyz)
    assert segs_df is None

    # Single segment with midpoint in voxel
    mock_sample.return_value.loc[
        segments.index[0],
        [*test_module.SEGMENT_START_COLS, *test_module.SEGMENT_END_COLS],
    ] = [10, 10, 10, 11, 11, 11]
    segs_df = test_module.pick_segments_voxel(circuit_path, 10 + min_xyz, 10 + max_xyz)
    assert len(segs_df) == 1
    assert {"segment_length", "section_type"} - set(segs_df.columns) == set()

    # check that we have the correct segment
    common_cols = list(set(segments.columns).intersection(segs_df.columns))
    assert all(segs_df[common_cols] == segments.loc[0][common_cols])

    # check that dataframe_cleanup is called if given
    mock_cleanup = Mock(return_value=None)
    segs_df = test_module.pick_segments_voxel(
        circuit_path, min_xyz, max_xyz, dataframe_cleanup=mock_cleanup
    )
    mock_cleanup.assert_called_once_with(segs_df)


def test_pick_synapses_locations():
    np.random.seed(666)
    count = 100
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    segments = fake_segments(min_xyz, max_xyz, 2)
    segments["segment_length"] = 1

    # equal probabilities for each segment
    mock_segment_pref = lambda x: np.ones(len(x))
    syns = test_module.pick_synapse_locations(segments, mock_segment_pref, count)
    assert len(syns) == count

    # check that we have syns from several segments
    assert len(syns[["section_id", "segment_id", "gid"]].drop_duplicates()) > 1

    # Locations should be different
    assert len(syns.drop_duplicates()) == count

    # segment_pref assigns prob 1 to one segment
    mock_segment_pref = lambda x: np.concatenate(([1], np.zeros(len(x) - 1)))
    syns = test_module.pick_synapse_locations(segments, mock_segment_pref, count)

    # should only have ids syns from one segment
    assert len(syns[["section_id", "segment_id", "gid"]].drop_duplicates()) == 1

    # segment_pref assigns prob 0 to every segment
    mock_segment_pref = lambda x: np.zeros(len(x))
    syns = test_module.pick_synapse_locations(segments, mock_segment_pref, count)
    assert syns is None


@patch.object(test_module, "pick_segments_voxel")
@patch.object(test_module, "pick_synapse_locations")
def test_pick_synapses_voxel(mock_pick_synapse_locations, mock_pick_segments_voxel):
    segments = fake_segments(np.array([0, 0, 0]), np.array([1, 1, 1]), 2)
    # make sure all wanted columns are present
    segments[test_module.WANTED_COLS] = 1

    mock_pick_segments_voxel.return_value = "not_none"
    mock_pick_synapse_locations.return_value = segments
    xyz_count = (0, 0, 1)
    circuit_path = "foo/bar/baz"

    res = test_module.pick_synapses_voxel(
        xyz_count, circuit_path, segment_pref=None, dataframe_cleanup=None
    )

    assert res is not None

    # check that unnecessary columns are removed
    assert len(res.columns) < len(segments.columns)
    assert all(res.eq(res[test_module.WANTED_COLS]))

    # pick_synapse_locations returns None
    mock_pick_synapse_locations.return_value = None
    res = test_module.pick_synapses_voxel(
        xyz_count, circuit_path, segment_pref=None, dataframe_cleanup=None
    )
    assert res is None

    # pick_segments_voxel returns None
    mock_pick_segments_voxel.return_value = None
    res = test_module.pick_synapses_voxel(
        xyz_count, circuit_path, segment_pref=None, dataframe_cleanup=None
    )
    assert res is None


@patch.object(test_module, "map_parallelize", new=lambda *args, **_: map(*args))
@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test_pick_synapses(mock_sample):
    count = 1250  # need many random test_module so sampling successfully finds enough
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([10, 10, 10])

    np.random.seed(0)
    mock_sample.return_value = fake_segments(min_xyz, max_xyz, 2 * count)

    # Fill xyz coordinates 3:7 with synapses. Each has 5.
    starts = np.array([*itertools.product(range(3, 7), repeat=3)])
    ends = starts + 1
    counts = np.full(len(starts), 5)
    xyzs_counts = np.hstack((starts, ends, counts[:, np.newaxis]))

    segs_df = test_module.pick_synapses("fake_path", xyzs_counts)

    # total count 4x4x4x5 = 320
    assert xyzs_counts[:, -1].sum() == len(segs_df) == 320
    assert set(segs_df.columns) == set(test_module.WANTED_COLS)


@patch.object(
    test_module,
    "map_parallelize",
    new=Mock(return_value=[pd.DataFrame({"fake": np.zeros(666)})]),
)
def test_pick_synapses_low_count(caplog):
    # total count 20x10x5 = 1000
    starts = np.array([*itertools.product(range(20), range(10), range(5))])
    ends = starts + 1
    counts = np.ones(len(starts))
    xyzs_counts = np.hstack((starts, ends, counts[:, np.newaxis]))

    with caplog.at_level(logging.WARNING):
        syns = test_module.pick_synapses("fake_path", xyzs_counts)
        assert len(syns) == 666
        assert "Could only pick 66.60 % of the intended synapses" in caplog.text


@patch.object(test_module, "pick_synapses_voxel", new=Mock(return_value=None))
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test_pick_synapses_chunk_all_return_none():
    xyz_counts = np.zeros((10, 7))
    index_path = "fake"
    res = test_module.pick_synapses_chunk(
        xyz_counts,
        index_path,
        segment_pref=None,
        dataframe_cleanup=None,
    )

    assert res is None


def test_build_synapses_default():
    height = VoxelData(np.arange(8).reshape((2, 2, 2)), (1, 1, 1))
    synapse_density = [[[0, 7], [2, 8], [3, 67], [7, 42]]]
    oversampling = 3
    syns = test_module.build_synapses_default(height, synapse_density, oversampling)
    assert_array_equal(syns.raw, [[[21, 21], [24, 201]], [[201, 201], [201, 0]]])

    # Test low density
    height = VoxelData(np.full((1000, 1, 1000), 1), (1, 1, 1))
    synapse_density = [[[0, 0.1], [2, 0.1]]]
    oversampling = 1
    syns = test_module.build_synapses_default(height, synapse_density, oversampling)
    assert_approx_equal(syns.raw.sum() / 1e6, 0.1, significant=3)


def test_organize_indices():
    syns = pd.DataFrame(
        [
            [2, 10],
            [1, 10],
            [2, 1],
            [1, 11],
            [2, 3],
        ],
        columns=["tgid", "sgid"],
    )
    ret = test_module.organize_indices(syns.copy())
    assert len(syns) == len(ret)
    assert np.all(np.diff(ret.tgid.values) >= 0)
