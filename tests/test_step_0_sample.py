import nose.tools as nt
import numpy as np
import numpy.testing as npt
import pandas as pd
from bluepy.v2.enums import Section, Segment
from mock import Mock, patch
from neurom import NeuriteType
from nose.tools import ok_
from numpy.testing import assert_equal
from voxcell import VoxelData

from projectionizer.synapses import (_min_max_axis, build_synapses_default,
                                     pick_synapses, pick_synapses_voxel,
                                     segment_pref_length)


def mock_segment_pref(segs_df):
    return np.ones(len(segs_df))


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


def _fake_voxel_synapse_count(shape, voxel_size=10):
    raw = np.zeros(shape=shape, dtype=np.int)
    raw[3:7, 3:7, 3:7] = 5
    return VoxelData(raw, [voxel_size] * 3, (0, 0, 0))


def test__min_max_axis():
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    min_, max_ = _min_max_axis(min_xyz, max_xyz)
    npt.assert_allclose(min_xyz, min_)
    npt.assert_allclose(max_xyz, max_)

    min_xyz = np.array([-10, -5, 1])
    max_xyz = np.array([-1, 0, -1])
    min_, max_ = _min_max_axis(min_xyz, max_xyz)
    npt.assert_allclose(min_, np.array([-10, -5, -1]))
    npt.assert_allclose(max_, np.array([-1, 0, 1]))


def test_pick_synapses_voxel():
    count = 10
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = 'foo/bar/baz'

    with patch('projectionizer.synapses.FI') as mock_FI, \
        patch('projectionizer.synapses.SegmentIndex') as mock_si:

        mock_si._wrap_result.return_value = _fake_segments(min_xyz, max_xyz, 2 * count)

        xyz_count = (min_xyz, max_xyz, count)
        segs_df = pick_synapses_voxel(xyz_count, circuit_path, mock_segment_pref)
        nt.eq_(count, len(segs_df))
        nt.ok_('x' in segs_df.columns)
        nt.ok_('segment_length' in segs_df.columns)

        segs_df = pick_synapses_voxel(xyz_count, circuit_path, lambda x: 0)
        ok_(segs_df is None)


@patch('projectionizer.synapses.map_parallelize', map)
def test_pick_synapses():
    count = 1250  # need many random synapses so sampling successfully finds enough
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = 'foo/bar/baz'

    np.random.seed(0)
    with patch('projectionizer.synapses.FI') as mock_FI, \
        patch('projectionizer.synapses.SegmentIndex') as mock_si:

        mock_si._wrap_result.return_value = _fake_segments(min_xyz, max_xyz, 2 * count)
        voxel_synapse_count = _fake_voxel_synapse_count(shape=(10, 10, 10), voxel_size=0.1)
        segs_df = pick_synapses(circuit_path, voxel_synapse_count, 100)

        nt.eq_(np.sum(voxel_synapse_count.raw), len(segs_df))
        nt.ok_('x' in segs_df.columns)
        nt.ok_('segment_length' in segs_df.columns)


def test_segment_pref():
    df = pd.DataFrame({Section.NEURITE_TYPE: [NeuriteType.axon,
                                              NeuriteType.axon,
                                              NeuriteType.basal_dendrite,
                                              NeuriteType.apical_dendrite,
                                              ],
                       'segment_length': 1})
    ret = segment_pref_length(df)
    ok_(isinstance(ret, pd.Series))
    assert_equal(ret.values, np.array([0., 0., 1., 1.]))


def test_build_synapses_default():
    height = VoxelData(np.arange(8).reshape((2, 2, 2)), (1, 1, 1))
    synapse_density = [[[0, 7], [2, 8], [3, 67], [7, 42]]]
    oversampling = 3
    synapses = build_synapses_default(height, synapse_density, oversampling)
    assert_equal(synapses.raw, [[[21,  21], [24, 201]], [[201, 201], [201, 0]]])
