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

    df = pd.DataFrame(index=np.arange(count), columns=COLUMNS)
    df[[Segment.X1, Segment.X2]] = (
        min_xyz[0] + (max_xyz[0] - min_xyz[0]) * np.random.random((2, count))).T
    df[[Segment.Y1, Segment.Y2]] = (
        min_xyz[1] + (max_xyz[1] - min_xyz[1]) * np.random.random((2, count))).T
    df[[Segment.Z1, Segment.Z2]] = (
        min_xyz[2] + (max_xyz[2] - min_xyz[2]) * np.random.random((2, count))).T
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

    circuit = Mock()
    circuit.morph.spatial_index.q_window.return_value = _fake_segments(min_xyz, max_xyz,
                                                                       2 * count)
    xyz_count = (min_xyz, max_xyz, count)
    segs_df = pick_synapses_voxel(xyz_count, circuit, mock_segment_pref)
    nt.eq_(count, len(segs_df))
    nt.ok_('x' in segs_df.columns)
    nt.ok_('segment_length' in segs_df.columns)

    segs_df = pick_synapses_voxel(xyz_count, circuit, lambda x: 0)


@patch('projectionizer.synapses.map_parallelize', map)
def test_pick_synapses():
    count = 10
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit = Mock()
    circuit.morph.spatial_index.q_window.return_value = _fake_segments(min_xyz,
                                                                       max_xyz,
                                                                       2 * count)
    voxel_synapse_count = _fake_voxel_synapse_count(shape=(10, 10, 10), voxel_size=0.1)
    pick_synapses(circuit, voxel_synapse_count, 100)

# def test_generate_ijk_counts():
#     min_ijk, max_ijk = (0, 0, 0), (10, 10, 10)
#     size = (10, 10, 10)
#     voxel_synapse_count = _fake_voxel_synapse_count(size)
#     itr = projection.generate_ijk_counts(min_ijk, max_ijk, voxel_synapse_count)
#     counts = list(itr)
#     ijk, min_xyz, max_xyz, count = counts[0]
#     npt.assert_equal(ijk, np.array([3, 3, 3]))
#     npt.assert_equal(min_xyz, np.array([25., 25., 25.]))
#     npt.assert_equal(max_xyz, np.array([35., 35., 35.]))
#     nt.eq_(count, 5)


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
