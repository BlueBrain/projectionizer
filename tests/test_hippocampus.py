from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from bluepy import Segment
from morphio import SectionType
from voxcell import VoxelData

import projectionizer
import projectionizer.hippocampus as test_module

from utils import fake_segments

VOXEL_DIMENSIONS = np.array([1, 1, 1])


def _run_full_sample_worker(positions):
    circuit_path = "foo/bar/baz"
    return test_module._full_sample_worker([positions], circuit_path, VOXEL_DIMENSIONS)


@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test__full_sample_worker(mock_sample):
    count = 10
    min_xyz = np.array([0, 0, 0])

    mock_sample.return_value = fake_segments(min_xyz, min_xyz + VOXEL_DIMENSIONS, count)

    res = _run_full_sample_worker(min_xyz)

    assert count == len(res)
    for col in test_module.SEGMENT_COLUMNS:
        assert col in res.columns

    # make sure locations are different
    assert len(res) == len(res.drop_duplicates())

    # no segments with their midpoint in the voxel
    position = np.array([10, 10, 10])
    res = _run_full_sample_worker(position)
    assert len(res) == 0


@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test__full_sample_worker_single_segment_in_voxel(mock_sample):
    count = 10
    min_xyz = np.array([0, 0, 0])

    # single segment with its midpoint in the voxel
    segments = fake_segments(min_xyz, min_xyz + VOXEL_DIMENSIONS, count)
    segments.loc[
        segments.index[0],
        [Segment.X1, Segment.Y1, Segment.Z1, Segment.X2, Segment.Y2, Segment.Z2],
    ] = [10, 10, 10, 11, 11, 11]
    mock_sample.return_value = segments

    position = np.array([10, 10, 10])
    res = _run_full_sample_worker(position)

    assert len(res) == 1

    for col in test_module.SEGMENT_COLUMNS:
        assert col in res.columns

    assert len(res) == len(res.drop_duplicates())

    # all get the same section/segment/gid, since only a single segment lies in the voxel
    assert len(res[["section_id", "segment_id", "tgid"]].drop_duplicates()) == 1


@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test__full_sample_worker_segments_axons(mock_sample):
    # all segments are axons
    count = 10
    min_xyz = np.array([0, 0, 0])

    segments = fake_segments(min_xyz, min_xyz + VOXEL_DIMENSIONS, count)
    segments["section_type"] = SectionType.axon
    mock_sample.return_value = segments

    res = _run_full_sample_worker(min_xyz)

    assert len(res) == 0


@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test__full_sample_worker_no_segments_returned(mock_sample):
    min_xyz = np.array([0, 0, 0])

    mock_sample.return_value = fake_segments(min_xyz, min_xyz + VOXEL_DIMENSIONS, 0)
    res = _run_full_sample_worker(min_xyz)

    assert len(res) == 0


# mock with a 'map' that ignores kwargs
@patch.object(projectionizer.utils, "map_parallelize", new=lambda *args, **_: map(*args))
@patch.object(test_module, "_sample_with_spatial_index")
@patch.object(test_module.spatial_index, "open_index", new=Mock())
def test_full_sample_parallel(mock_sample, tmp_confdir):
    count = 50
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = "foo/bar/baz"
    REGION = "TEST"
    REGION_ID = 1
    brain_regions = VoxelData(np.array([[[REGION_ID, 0], [0, 0]]]), np.array([1, 1, 1]))

    mock_sample.return_value = fake_segments(min_xyz, max_xyz, count)

    # test that normally, file is created an it has expected data
    test_module.full_sample_parallel(brain_regions, REGION, REGION_ID, circuit_path, tmp_confdir)

    feather_path = tmp_confdir / f"{REGION}_{REGION_ID}_000.feather"
    assert feather_path.is_file()
    segs_df = pd.read_feather(feather_path)

    assert len(segs_df) == count
    for col in test_module.SEGMENT_COLUMNS:
        assert col in segs_df.columns


@patch.object(test_module, "_full_sample_worker")
@patch.object(projectionizer.utils, "map_parallelize", new=lambda *args, **_: map(*args))
def test_full_sample_parallel_skip(mock_worker, tmp_confdir):
    circuit_path = "foo/bar/baz"
    REGION = "TEST"
    REGION_ID = 1
    FAKE_ID = 666

    brain_regions = VoxelData(np.array([[[REGION_ID, 0], [0, 0]]]), np.array([1, 1, 1]))

    # if region is not in brain_regions, test no file is created nor sampling is done
    test_module.full_sample_parallel(brain_regions, REGION, FAKE_ID, circuit_path, tmp_confdir)

    fake_feather_path = tmp_confdir / f"{REGION}_{FAKE_ID}_000.feather"
    assert not fake_feather_path.exists()
    mock_worker.assert_not_called()

    # assert that no sampling is done if file exists
    feather_path = tmp_confdir / f"{REGION}_{REGION_ID}_000.feather"
    feather_path.write_text("")
    test_module.full_sample_parallel(brain_regions, REGION, REGION_ID, circuit_path, tmp_confdir)
    mock_worker.assert_not_called()
