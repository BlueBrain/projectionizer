import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from bluepy import Segment
from morphio import SectionType
from numpy.testing import (
    assert_approx_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from voxcell import VoxelData

import projectionizer.synapses as test_module

from utils import fake_segments


def _fake_voxel_synapse_count(shape, voxel_size=10):
    raw = np.zeros(shape=shape, dtype=int)
    return VoxelData(raw, [voxel_size] * 3, (0, 0, 0))


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

    start_pos = [Segment.X1, Segment.Y1, Segment.Z1]
    end_pos = [Segment.X2, Segment.Y2, Segment.Z2]

    # Segment starts == Segment ends except for one entry:
    segments[end_pos] = np.full((np.shape(segments)[0], 3), 1)
    segments[start_pos] = segments[end_pos]
    segments.loc[segments.index[1], Segment.X1] += 1
    segments.loc[segments.index[1], "gid"] = 666

    mock_sample.return_value = segments

    res = test_module.spherical_sampling((np.array([0, 0, 0]), 1), "fake_path", radius=5)
    assert_array_equal(res.gid, segments.iloc[1].gid)


@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test_pick_synapses_voxel(mock_sample):
    count = 10
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = "foo/bar/baz"

    mock_sample.return_value = fake_segments(min_xyz, max_xyz, 2 * count)

    def mock_segment_pref(segs_df):
        return np.ones(len(segs_df))

    # ask for duplicates, to make sure locations are different, even though
    # the same segment is being used
    count *= 10

    xyz_count = min_xyz, max_xyz, count
    segs_df = test_module.pick_synapses_voxel(
        xyz_count,
        circuit_path,
        mock_segment_pref,
        dataframe_cleanup=None,
    )
    assert count == len(segs_df)
    assert "x" in segs_df.columns
    assert "segment_length" in segs_df.columns

    # make sure locations are different
    assert len(segs_df) == len(segs_df.drop_duplicates())

    # no segments with their midpoint in the voxel
    xyz_count = np.array([10, 10, 10]), np.array([11, 11, 11]), count
    segs_df = test_module.pick_synapses_voxel(
        xyz_count, circuit_path, mock_segment_pref, dataframe_cleanup=None
    )
    assert segs_df is None

    # single segment with its midpoint in the voxel
    segments = fake_segments(min_xyz, max_xyz, 2 * count)
    segments.loc[
        segments.index[0],
        [Segment.X1, Segment.Y1, Segment.Z1, Segment.X2, Segment.Y2, Segment.Z2],
    ] = [10, 10, 10, 11, 11, 11]

    mock_sample.return_value = segments

    segs_df = test_module.pick_synapses_voxel(
        xyz_count,
        circuit_path,
        mock_segment_pref,
        dataframe_cleanup=None,
    )
    assert count == len(segs_df)
    assert "x" in segs_df.columns
    assert "segment_length" in segs_df.columns
    assert len(segs_df) == len(segs_df.drop_duplicates())

    # all get the same section/segment/gid, since only a single segment lies in the voxel
    assert len(segs_df[["section_id", "segment_id", "gid"]].drop_duplicates()) == 1

    # segment_pref picks no test_module
    segs_df = test_module.pick_synapses_voxel(
        xyz_count,
        circuit_path,
        lambda x: 0,
        dataframe_cleanup=None,
    )
    assert segs_df is None

    # return None if libFLATindex finds nothing
    mock_sample.return_value = None
    segs_df = test_module.pick_synapses_voxel(
        xyz_count,
        circuit_path,
        mock_segment_pref,
        dataframe_cleanup=None,
    )
    assert segs_df is None


@patch.object(test_module, "map_parallelize", new=lambda *args, **_: map(*args))
@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test_pick_synapses(mock_sample):
    count = 1250  # need many random test_module so sampling successfully finds enough
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = "foo/bar/baz"

    np.random.seed(0)
    mock_sample.return_value = fake_segments(min_xyz, max_xyz, 2 * count)
    voxel_synapse_count = _fake_voxel_synapse_count(shape=(10, 10, 10), voxel_size=0.1)
    # total count 4x4x4x5 = 320
    voxel_synapse_count.raw[3:7, 3:7, 3:7] = 5
    segs_df = test_module.pick_synapses(circuit_path, voxel_synapse_count)

    assert np.sum(voxel_synapse_count.raw) == len(segs_df) == 320
    assert "x" in segs_df.columns
    assert "segment_length" in segs_df.columns


@patch.object(
    test_module,
    "map_parallelize",
    new=Mock(return_value=[pd.DataFrame({"fake": np.zeros(666)})]),
)
def test_pick_synapses_low_count(caplog):
    # total count 20x10x5 = 1000
    voxel_synapse_count = _fake_voxel_synapse_count(shape=(20, 10, 1))
    voxel_synapse_count.raw[:, :, :] = 5

    with caplog.at_level(logging.WARNING):
        syns = test_module.pick_synapses("fake_path", voxel_synapse_count)
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
