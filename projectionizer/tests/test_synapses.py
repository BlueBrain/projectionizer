import numpy as np
import pandas as pd
from bluepy import Section, Segment
from mock import patch
from neurom import NeuriteType
from numpy.testing import (
    assert_approx_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from voxcell import VoxelData

from projectionizer import synapses

from utils import fake_segments


def test_segment_pref_length():
    df = pd.DataFrame({Section.NEURITE_TYPE: [NeuriteType.axon,
                                              NeuriteType.axon,
                                              NeuriteType.basal_dendrite,
                                              NeuriteType.apical_dendrite,
                                              ],
                       'segment_length': 1})
    ret = synapses.segment_pref_length(df)
    assert isinstance(ret, pd.Series)
    assert_array_equal(ret.values, [0, 0, 1, 1])


def test_get_segment_limits_within_sphere():
    a = np.array([[0, 10, -10],
                  [0, 6, 6],
                  [0, 1, 1],
                  [0, -10, -10],
                  [-2, -2, -2]], dtype=float)
    b = np.array([[0, 10, 10],
                  [0, 10, 10],
                  [0, 10, 10],
                  [0, 10, 10],
                  [2, 2, 2]], dtype=float)
    point = np.array((0, 0, 0))

    res_start, res_end = synapses.get_segment_limits_within_sphere(a, b, point, radius=5)

    expected_start = np.array([[np.nan, np.nan, np.nan],
                               [np.nan, np.nan, np.nan],
                               [0, 1, 1],
                               [0, -np.sqrt(25 / 2), -np.sqrt(25 / 2)],
                               [-2, -2, -2]])

    expected_end = np.array([[np.nan, np.nan, np.nan],
                             [np.nan, np.nan, np.nan],
                             [0, np.sqrt(25 / 2), np.sqrt(25 / 2)],
                             [0, np.sqrt(25 / 2), np.sqrt(25 / 2)],
                             [2, 2, 2]])

    assert_array_almost_equal(res_start, expected_start)
    assert_array_almost_equal(res_end, expected_end)

    for _ in range(5):
        addition = np.random.random((1, 3)) * np.random.randint(10, 100, (1, 3))
        non_zero_start, non_zero_end = synapses.get_segment_limits_within_sphere(a + addition,
                                                                                 b + addition,
                                                                                 point + addition,
                                                                                 radius=5)
        assert_array_almost_equal(non_zero_start, expected_start + addition)
        assert_array_almost_equal(non_zero_end, expected_end + addition)

    # Test that results are same if ends and starts are reversed
    rev_start, rev_end = synapses.get_segment_limits_within_sphere(b, a, point, radius=5)
    assert_array_almost_equal(rev_start, expected_end)
    assert_array_almost_equal(rev_end, expected_start)


def test_spherical_sampling():

    min_xyz = np.array([10, 10, 10])
    max_xyz = min_xyz + 1

    segments = fake_segments(min_xyz, max_xyz, 5)

    with patch('projectionizer.synapses._sample_with_flat_index') as mock_sample:
        mock_sample.return_value = segments

        with patch('projectionizer.synapses.get_segment_limits_within_sphere') as mock_points:
            return_value = np.full((2, 5, 3), np.nan)
            return_value[0, :2, :] = 1
            return_value[1, :2, :] = 2
            mock_points.return_value = return_value
            res = synapses.spherical_sampling((np.array([0, 0, 0]), 1), 'fake_path', radius=5)

            assert len(res) == 2
            assert_array_equal(res.gid, segments.iloc[:2].gid)

    # Test pruning of zero length segments
    start_pos = [Segment.X1, Segment.Y1, Segment.Z1]
    end_pos = [Segment.X2, Segment.Y2, Segment.Z2]
    segments[end_pos] = np.full((np.shape(segments)[0], 3), 1)
    segments[start_pos] = segments[end_pos]
    segments.loc[segments.index[1], Segment.X1] += 1
    segments.loc[segments.index[1], 'gid'] = 666

    with patch('projectionizer.synapses._sample_with_flat_index') as mock_sample:
        mock_sample.return_value = segments
        res = synapses.spherical_sampling((np.array([0, 0, 0]), 1), 'fake_path', radius=5)
        assert_array_equal(res.gid, segments.iloc[1].gid)


def test_pick_synapses_voxel():
    count = 10
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = 'foo/bar/baz'

    def mock_segment_pref(segs_df):
        return np.ones(len(segs_df))

    with patch('projectionizer.synapses._sample_with_flat_index') as mock_sample:
        mock_sample.return_value = fake_segments(min_xyz, max_xyz, 2 * count)

        # ask for duplicates, to make sure locations are different, eventhough
        # the same segment is being used
        count *= 10

        xyz_count = min_xyz, max_xyz, count
        segs_df = synapses.pick_synapses_voxel(xyz_count,
                                               circuit_path,
                                               mock_segment_pref,
                                               dataframe_cleanup=None
                                               )
        assert count == len(segs_df)
        assert 'x' in segs_df.columns
        assert 'segment_length' in segs_df.columns

        # make sure locations are different
        assert len(segs_df) == len(segs_df.drop_duplicates())

        # no segments with their midpoint in the voxel
        xyz_count = np.array([10, 10, 10]), np.array([11, 11, 11]), count
        segs_df = synapses.pick_synapses_voxel(xyz_count,
                                               circuit_path,
                                               mock_segment_pref,
                                               dataframe_cleanup=None
                                               )
        assert segs_df is None

        # single segment with its midpoint in the voxel
        segments = fake_segments(min_xyz, max_xyz, 2 * count)
        segments.loc[segments.index[0], [Segment.X1, Segment.Y1, Segment.Z1,
                                         Segment.X2, Segment.Y2, Segment.Z2]] = [10, 10, 10,
                                                                                 11, 11, 11]

        mock_sample.return_value = segments

        segs_df = synapses.pick_synapses_voxel(xyz_count,
                                               circuit_path,
                                               mock_segment_pref,
                                               dataframe_cleanup=None
                                               )
        assert count == len(segs_df)
        assert 'x' in segs_df.columns
        assert 'segment_length' in segs_df.columns
        assert len(segs_df) == len(segs_df.drop_duplicates())

        # all get the same section/segment/gid, since only a single segment lies in the voxel
        assert len(segs_df[[Section.ID, Segment.ID, 'gid']].drop_duplicates()) == 1

        # segment_pref picks no synapses
        segs_df = synapses.pick_synapses_voxel(xyz_count,
                                               circuit_path,
                                               lambda x: 0,
                                               dataframe_cleanup=None)
        assert segs_df is None

        # return None if libFLATindex finds nothing
        mock_sample.return_value = None
        segs_df = synapses.pick_synapses_voxel(xyz_count,
                                               circuit_path,
                                               mock_segment_pref,
                                               dataframe_cleanup=None
                                               )
        assert segs_df is None


def test_pick_synapses():
    count = 1250  # need many random synapses so sampling successfully finds enough
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = 'foo/bar/baz'

    def _fake_voxel_synapse_count(shape, voxel_size=10):
        raw = np.zeros(shape=shape, dtype=int)
        raw[3:7, 3:7, 3:7] = 5
        return VoxelData(raw, [voxel_size] * 3, (0, 0, 0))

    np.random.seed(0)
    with patch('projectionizer.synapses._sample_with_flat_index') as mock_sample, \
            patch('projectionizer.synapses.map_parallelize', map):
        mock_sample.return_value = fake_segments(min_xyz, max_xyz, 2 * count)
        voxel_synapse_count = _fake_voxel_synapse_count(shape=(10, 10, 10), voxel_size=0.1)
        segs_df = synapses.pick_synapses(circuit_path, voxel_synapse_count)

        assert np.sum(voxel_synapse_count.raw) == len(segs_df)
        assert 'x' in segs_df.columns
        assert 'segment_length' in segs_df.columns


def test_build_synapses_default():
    height = VoxelData(np.arange(8).reshape((2, 2, 2)), (1, 1, 1))
    synapse_density = [[[0, 7], [2, 8], [3, 67], [7, 42]]]
    oversampling = 3
    syns = synapses.build_synapses_default(height, synapse_density, oversampling)
    assert_array_equal(syns.raw, [[[21, 21], [24, 201]], [[201, 201], [201, 0]]])

    # Test low density
    height = VoxelData(np.full((1000, 1, 1000), 1), (1, 1, 1))
    synapse_density = [[[0, .1], [2, .1]]]
    oversampling = 1
    syns = synapses.build_synapses_default(height, synapse_density, oversampling)
    assert_approx_equal(syns.raw.sum() / 1e6, .1, significant=3)


def test_organize_indices():
    syns = pd.DataFrame([[2, 10],
                         [1, 10],
                         [2, 1],
                         [1, 11],
                         [2, 3],
                         ],
                        columns=['tgid', 'sgid'])
    ret = synapses.organize_indices(syns.copy())
    assert len(syns) == len(ret)
    assert np.all(np.diff(ret.tgid.values) >= 0)
