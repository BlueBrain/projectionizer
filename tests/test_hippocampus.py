import os

import numpy as np
import pandas as pd
from bluepy import Section, Segment
from mock import Mock, patch
from neurom import NeuriteType
from voxcell import VoxelData

from projectionizer import hippocampus as test_module

from utils import fake_segments, setup_tempdir


def test__full_sample_worker():
    count = 10
    min_xyz = np.array([0, 0, 0])
    voxel_dimensions = np.array([1, 1, 1])
    circuit_path = "foo/bar/baz"

    with patch("projectionizer.synapses._sample_with_flat_index") as mock_sample:
        mock_sample.return_value = fake_segments(min_xyz, min_xyz + voxel_dimensions, count)

        segs_df = test_module._full_sample_worker(
            [min_xyz],
            circuit_path,
            voxel_dimensions,
        )
        assert count == len(segs_df)
        for col in test_module.SEGMENT_COLUMNS:
            assert col in segs_df.columns

        # make sure locations are different
        assert len(segs_df) == len(segs_df.drop_duplicates())

        # no segments with their midpoint in the voxel
        new_min_xyz = np.array([10, 10, 10])
        segs_df = test_module._full_sample_worker(
            [new_min_xyz],
            circuit_path,
            voxel_dimensions,
        )
        assert len(segs_df) == 0

        # single segment with its midpoint in the voxel
        segments = fake_segments(min_xyz, min_xyz + voxel_dimensions, count)
        segments.loc[
            segments.index[0],
            [Segment.X1, Segment.Y1, Segment.Z1, Segment.X2, Segment.Y2, Segment.Z2],
        ] = [10, 10, 10, 11, 11, 11]
        mock_sample.return_value = segments

        segs_df = test_module._full_sample_worker(
            [new_min_xyz],
            circuit_path,
            voxel_dimensions,
        )
        assert len(segs_df) == 1

        for col in test_module.SEGMENT_COLUMNS:
            assert col in segs_df.columns

        assert len(segs_df) == len(segs_df.drop_duplicates())

        # all get the same section/segment/gid, since only a single segment lies in the voxel
        assert len(segs_df[["section_id", "segment_id", "tgid"]].drop_duplicates()) == 1

        # all segments are axons
        segments = fake_segments(min_xyz, min_xyz + voxel_dimensions, count)
        segments[Section.NEURITE_TYPE] = NeuriteType.axon
        mock_sample.return_value = segments
        segs_df = test_module._full_sample_worker(
            [new_min_xyz],
            circuit_path,
            voxel_dimensions,
        )
        assert len(segs_df) == 0

        # no segments returned
        mock_sample.return_value = fake_segments(min_xyz, min_xyz + voxel_dimensions, 0)
        segs_df = test_module._full_sample_worker(
            [new_min_xyz],
            circuit_path,
            voxel_dimensions,
        )
        assert len(segs_df) == 0


def test_full_sample_parallel():
    count = 50
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    circuit_path = "foo/bar/baz"
    REGION = "TEST"
    REGION_ID = 1
    FAKE_ID = 666
    brain_regions = VoxelData(np.array([[[REGION_ID, 0], [0, 0]]]), np.array([1, 1, 1]))

    # map takes no kwargs
    def kwarg_map(*args, **_):
        return map(*args)

    with patch("projectionizer.synapses._sample_with_flat_index") as mock_sample, patch(
        "projectionizer.utils.map_parallelize", kwarg_map
    ):
        mock_sample.return_value = fake_segments(min_xyz, max_xyz, count)
        with setup_tempdir("test_hippocampus") as outdir:
            # test that normally, file is created an it has expected data
            test_module.full_sample_parallel(brain_regions, REGION, REGION_ID, circuit_path, outdir)

            sample_path = os.path.join(outdir, test_module.SAMPLE_PATH)
            assert os.path.isdir(sample_path)

            feather_path = os.path.join(sample_path, f"{REGION}_{REGION_ID}_000.feather")
            assert os.path.isfile(feather_path)
            segs_df = pd.read_feather(feather_path)

            assert len(segs_df) == count
            for col in test_module.SEGMENT_COLUMNS:
                assert col in segs_df.columns

            mock_worker = Mock()
            with patch("projectionizer.hippocampus._full_sample_worker", mock_worker):
                # if region is not in brain_regions, test no file is created nor sampling is done
                test_module.full_sample_parallel(
                    brain_regions, REGION, FAKE_ID, circuit_path, outdir
                )

                feather_path = os.path.join(sample_path, f"{REGION}_{FAKE_ID}_000.feather")
                assert not os.path.isfile(feather_path)
                mock_worker.assert_not_called()

                # assert that no sampling is done if file exists
                test_module.full_sample_parallel(
                    brain_regions, REGION, REGION_ID, circuit_path, outdir
                )
                mock_worker.assert_not_called()
