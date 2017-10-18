import nose.tools as nt
from nose.tools import ok_, eq_, raises
from mock import Mock

import numpy as np
import pandas as pd
import numpy.testing as npt
from bluepy.v2.enums import Section, Segment
from neurom import NeuriteType
from voxcell import VoxelData

from projectionizer import projection


def _fake_segments(min_xyz, max_xyz, count):
    RADIUS = 10
    COLUMNS = [Segment.X1, Segment.Y1, Segment.Z1,
               Segment.X2, Segment.Y2, Segment.Z2,
               Segment.R1, Segment.R2, u'gid',
               Section.ID, Segment.ID, Section.NEURITE_TYPE]

    df = pd.DataFrame(index=np.arange(count), columns=COLUMNS)
    df[[Segment.X1, Segment.X2]] = (min_xyz[0] + (max_xyz[0] - min_xyz[0]) * np.random.random((2, count))).T
    df[[Segment.Y1, Segment.Y2]] = (min_xyz[1] + (max_xyz[1] - min_xyz[1]) * np.random.random((2, count))).T
    df[[Segment.Z1, Segment.Z2]] = (min_xyz[2] + (max_xyz[2] - min_xyz[2]) * np.random.random((2, count))).T
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
    min_, max_ = projection._min_max_axis(min_xyz, max_xyz)
    npt.assert_allclose(min_xyz, min_)
    npt.assert_allclose(max_xyz, max_)

    min_xyz = np.array([-10, -5, 1])
    max_xyz = np.array([-1, 0, -1])
    min_, max_ = projection._min_max_axis(min_xyz, max_xyz)
    npt.assert_allclose(min_, np.array([-10, -5, -1]))
    npt.assert_allclose(max_, np.array([-1, 0, 1]))


def test_pick_synapses_voxel():
    count = 10
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])

    circuit = Mock()
    circuit.morph.spatial_index.q_window.return_value = _fake_segments(min_xyz, max_xyz, 2*count)

    def segment_pref(segs_df):
        return np.ones(len(segs_df))

    segs_df = projection.pick_synapses_voxel(circuit, min_xyz, max_xyz, count, segment_pref)
    nt.eq_(count, len(segs_df))
    nt.ok_('x' in segs_df.columns)
    nt.ok_('segment_length' in segs_df.columns)


def test_generate_ijk_counts():
    min_ijk, max_ijk = (0, 0, 0), (10, 10, 10)
    size = (10, 10, 10)
    voxel_synapse_count = _fake_voxel_synapse_count(size)
    itr = projection.generate_ijk_counts(min_ijk, max_ijk, voxel_synapse_count)
    counts = list(itr)
    ijk, min_xyz, max_xyz, count = counts[0]
    npt.assert_equal(ijk, np.array([3, 3, 3]))
    npt.assert_equal(min_xyz, np.array([25., 25., 25.]))
    npt.assert_equal(max_xyz, np.array([35., 35., 35.]))
    nt.eq_(count, 5)


#def test_pick_synapses():
#    projection.pick_synapses(circuit, voxel_synapse_count, min_ijk, max_ijk, segment_pref)
