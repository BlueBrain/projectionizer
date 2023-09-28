from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import spatial_index
from voxcell import VoxelData

import projectionizer
import projectionizer.hippocampus as test_module

from utils import fake_segments

VOXEL_DIMENSIONS = np.array([1, 1, 1])


def _run_full_sample_worker(positions):
    circuit_path = "foo/bar/baz"
    return test_module._full_sample_worker([positions], circuit_path, VOXEL_DIMENSIONS)


@patch.object(spatial_index, "open_index", new=Mock())
@patch.object(projectionizer.synapses, "pick_segments_voxel")
def test__full_sample_worker(mock_sample):
    count = 10
    min_xyz = np.array([0, 0, 0])

    segs = fake_segments(min_xyz, min_xyz + VOXEL_DIMENSIONS, count)
    segs["section_type"] = 10
    segs["segment_length"] = 10
    mock_sample.return_value = segs[test_module.SEGMENT_COLUMNS]

    res = _run_full_sample_worker(min_xyz)

    assert count == len(res)
    assert set(res.columns) == set(test_module.SEGMENT_COLUMNS)


@patch.object(spatial_index, "open_index", new=Mock())
@patch.object(projectionizer.synapses, "pick_segments_voxel")
def test__full_sample_worker_no_segments_returned(mock_sample):
    min_xyz = np.array([0, 0, 0])
    mock_sample.return_value = None
    res = _run_full_sample_worker(min_xyz)
    assert set(res.columns) == set(test_module.SEGMENT_COLUMNS)
    assert len(res) == 0


# mock with a 'map' that ignores kwargs
@patch.object(spatial_index, "open_index", new=Mock())
@patch.object(projectionizer.utils, "map_parallelize", new=lambda *args, **_: map(*args))
@patch.object(projectionizer.synapses, "pick_segments_voxel")
def test_full_sample_parallel(mock_sample, tmp_confdir):
    count = 50
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = "foo/bar/baz"
    REGION = "TEST"
    REGION_ID = 1
    brain_regions = VoxelData(np.array([[[REGION_ID, 0], [0, 0]]]), np.array([1, 1, 1]))

    segs = fake_segments(min_xyz, max_xyz, count)
    segs["section_type"] = 10
    segs["segment_length"] = 10
    mock_sample.return_value = segs[test_module.SEGMENT_COLUMNS]

    # test that normally, file is created an it has expected data
    test_module.full_sample_parallel(brain_regions, REGION, REGION_ID, circuit_path, tmp_confdir)

    feather_path = tmp_confdir / f"{REGION}_{REGION_ID}_000.feather"
    assert feather_path.is_file()
    segs_df = pd.read_feather(feather_path)

    assert len(segs_df) == count

    # expect `gid` to be renamed to `tgid`
    expected_cols = (set(test_module.SEGMENT_COLUMNS) - {"gid"}) | {"tgid"}
    assert set(segs_df.columns) == set(expected_cols)


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
