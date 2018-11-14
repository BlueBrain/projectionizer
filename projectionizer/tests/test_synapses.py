from bluepy.v2.enums import Section, Segment
from mock import Mock, patch
from neurom import NeuriteType
from nose.tools import ok_, eq_
from numpy.testing import assert_equal, assert_allclose
from projectionizer import synapses
import numpy as np
import pandas as pd
from voxcell import VoxelData


def test_segment_pref():
    df = pd.DataFrame({Section.NEURITE_TYPE: [NeuriteType.axon,
                                              NeuriteType.axon,
                                              NeuriteType.basal_dendrite,
                                              NeuriteType.apical_dendrite,
                                              ],
                       'segment_length': 1})
    ret = synapses.segment_pref_length(df)
    ok_(isinstance(ret, pd.Series))
    assert_equal(ret.values, np.array([0., 0., 1., 1.]))


def _fake_segments(min_xyz, max_xyz, count):
    RADIUS = 10
    COLUMNS = [Segment.X1, Segment.Y1, Segment.Z1,
               Segment.X2, Segment.Y2, Segment.Z2,
               Segment.R1, Segment.R2, u'gid',
               Section.ID, Segment.ID, Section.NEURITE_TYPE]

    def samp(ax):
        return (min_xyz[ax] + (max_xyz[ax] - min_xyz[ax]) * np.random.random((2, count))).T

    X, Y, Z = 0, 1, 2
    df = pd.DataFrame(index=np.arange(count), columns=COLUMNS)
    df[[Segment.X1, Segment.X2]] = samp(X)
    df[[Segment.Y1, Segment.Y2]] = samp(Y)
    df[[Segment.Z1, Segment.Z2]] = samp(Z)
    df[[Segment.R1, Segment.R2]] = (RADIUS * np.random.random((2, count))).T

    df[[Section.ID, Segment.ID, 'gid']] = np.random.randint(100, size=(3, count)).T
    df[Section.NEURITE_TYPE] = NeuriteType.apical_dendrite

    return df


def test_pick_synapses_voxel():
    count = 10
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = 'foo/bar/baz'

    def mock_segment_pref(segs_df):
        return np.ones(len(segs_df))

    with patch('projectionizer.synapses._sample_with_flat_index') as mock_sample:
        mock_sample.return_value = _fake_segments(min_xyz, max_xyz, 2 * count)

        xyz_count = (min_xyz, max_xyz, count)
        segs_df = synapses.pick_synapses_voxel(xyz_count,
                                               circuit_path,
                                               mock_segment_pref,
                                               dataframe_cleanup=None
                                               )
        eq_(count, len(segs_df))
        ok_('x' in segs_df.columns)
        ok_('segment_length' in segs_df.columns)

    #with patch('projectionizer.synapses.FI') as mock_FI, \
    #    patch('projectionizer.synapses.SegmentIndex') as mock_si:

    #    mock_si._wrap_result.return_value = _fake_segments(min_xyz, max_xyz, 2 * count)

    #    xyz_count = (min_xyz, max_xyz, count)
    #    segs_df = synapses.pick_synapses_voxel(xyz_count, circuit_path, mock_segment_pref)
    #    eq_(count, len(segs_df))
    #    ok_('x' in segs_df.columns)
    #    ok_('segment_length' in segs_df.columns)

    #    # segment_pref picks no synapses
    #    segs_df = synapses.pick_synapses_voxel(xyz_count, circuit_path, lambda x: 0)
    #    ok_(segs_df is None)

    #    # no segments found
    #    mock_si._wrap_result.return_value = _fake_segments(min_xyz, max_xyz, count=0)
    #    segs_df = synapses.pick_synapses_voxel(xyz_count, circuit_path, lambda x: 0)
    #    ok_(segs_df is None)

    #    # FI throws exception
    #    mock_FI.numpy_windowQuery.side_effect = Exception()
    #    segs_df = synapses.pick_synapses_voxel(xyz_count, circuit_path, lambda x: 0)
    #    ok_(segs_df is None)


def test__min_max_axis():
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    min_, max_ = synapses._min_max_axis(min_xyz, max_xyz)
    assert_allclose(min_xyz, min_)
    assert_allclose(max_xyz, max_)

    min_xyz = np.array([-10, -5, 1])
    max_xyz = np.array([-1, 0, -1])
    min_, max_ = synapses._min_max_axis(min_xyz, max_xyz)
    assert_allclose(min_, np.array([-10, -5, -1]))
    assert_allclose(max_, np.array([-1, 0, 1]))


def test_pick_synapses():
    count = 1250  # need many random synapses so sampling successfully finds enough
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = 'foo/bar/baz'

    def _fake_voxel_synapse_count(shape, voxel_size=10):
        raw = np.zeros(shape=shape, dtype=np.int)
        raw[3:7, 3:7, 3:7] = 5
        return VoxelData(raw, [voxel_size] * 3, (0, 0, 0))

    np.random.seed(0)
    with patch('projectionizer.synapses._sample_with_flat_index') as mock_sample, \
        patch('projectionizer.synapses.map_parallelize', map):
        mock_sample.return_value = _fake_segments(min_xyz, max_xyz, 2 * count)
        voxel_synapse_count = _fake_voxel_synapse_count(shape=(10, 10, 10), voxel_size=0.1)
        segs_df = synapses.pick_synapses(circuit_path, voxel_synapse_count)

        eq_(np.sum(voxel_synapse_count.raw), len(segs_df))
        ok_('x' in segs_df.columns)
        ok_('segment_length' in segs_df.columns)


def test_build_synapses_default():
    height = VoxelData(np.arange(8).reshape((2, 2, 2)), (1, 1, 1))
    synapse_density = [[[0, 7], [2, 8], [3, 67], [7, 42]]]
    oversampling = 3
    syns = synapses.build_synapses_default(height, synapse_density, oversampling)
    assert_equal(syns.raw, [[[21,  21], [24, 201]], [[201, 201], [201, 0]]])


def test_organize_indices():
    syns = pd.DataFrame([[2, 10],
                         [1, 10],
                         [2, 1],
                         [1, 11],
                         [2, 3],
                         ],
                        columns=['tgid', 'sgid'])
    ret = synapses.organize_indices(syns.copy())
    eq_(len(syns), len(ret))
    ok_(np.all(0 <= np.diff(ret.tgid.values)))
    ok_(all(df.afferent_indices.max() == len(df) - 1 for _, df in ret.groupby('tgid')))
