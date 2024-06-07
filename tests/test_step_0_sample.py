from unittest.mock import Mock, patch

import brain_indexer.experimental
import luigi
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from voxcell import VoxelData

import projectionizer
import projectionizer.step_0_sample as test_module


def _synapse_count():
    raw = np.zeros((20, 20, 20))
    raw[5:15, 5:15, 5:15] = 1
    return VoxelData(raw, [1, 1, 1])


@patch.object(brain_indexer.experimental, "space_filling_order")
@pytest.mark.MockTask(cls=test_module.VoxelOrder)
def test_VoxelOrder(mock_space_filling_order, MockTask):
    voxel_counts = _synapse_count()
    voxel_synapse_count_path = MockTask.folder / "voxel-synapse-count.nrrd"
    voxel_counts.save_nrrd(voxel_synapse_count_path)

    np.random.seed(666)
    random_order = np.random.permutation((voxel_counts.raw != 0).sum())
    mock_space_filling_order.return_value = random_order

    class TestVoxelOrder(MockTask):
        def input(self):
            return Mock(path=voxel_synapse_count_path)

    task = TestVoxelOrder()
    task.run()

    # Check that indices are written in expected order
    res = projectionizer.utils.load(task.output().path)
    expected = np.transpose(np.nonzero(voxel_counts.raw))[random_order]
    npt.assert_array_equal(res.to_numpy(), expected)


@pytest.mark.MockTask(cls=test_module.SampleChunk)
def test__get_voxel_indices(MockTask):
    voxel_order_path = MockTask.folder / "voxel-order.feather"
    voxel_order = pd.DataFrame({col: np.arange(666) for col in projectionizer.utils.XYZ})
    projectionizer.utils.write_feather(voxel_order_path, voxel_order)
    n_chunks = 5

    class TestSampleChunk(MockTask):
        n_total_chunks = n_chunks
        chunk_num = luigi.IntParameter()

        def input(self):
            return Mock(path=voxel_order_path), None

    chunks = [TestSampleChunk(chunk_num=i)._get_voxel_indices() for i in range(n_chunks)]
    res = np.concatenate(chunks)
    npt.assert_array_equal(res, voxel_order.to_numpy())

    len_chunks = [*map(len, chunks)]
    npt.assert_array_equal(len_chunks, [134, 134, 134, 134, 130])


@pytest.mark.MockTask(cls=test_module.SampleChunk)
def test__get_xyzs_count(MockTask):
    # Create `voxel-synapse-count.nrrd` and `voxel-order.feather` so, that when synapse counts
    # are read in `voxel-order`, they should be [0, 1, 2, ...]
    voxel_counts = _synapse_count()
    mask = voxel_counts.raw != 0

    # Set counts [0, 1, 2, ...] in a random order
    np.random.seed(666)
    random_order = np.random.permutation(mask.sum())
    voxel_counts.raw[mask] = random_order
    voxel_synapse_count_path = MockTask.folder / "voxel-synapse-count.nrrd"
    voxel_counts.save_nrrd(voxel_synapse_count_path)

    # Set voxel order so that that the counts should be ordered if read in that order
    correct_order = np.argsort(random_order)
    voxel_order = np.array(np.where(mask)).T[correct_order]
    voxel_order = pd.DataFrame(voxel_order, columns=projectionizer.utils.XYZ)
    voxel_order_path = MockTask.folder / "voxel-order.feather"
    projectionizer.utils.write_feather(voxel_order_path, voxel_order)

    n_chunks = 5

    class TestSampleChunk(MockTask):
        n_total_chunks = n_chunks
        chunk_num = luigi.IntParameter()

        def input(self):
            return Mock(path=voxel_order_path), Mock(path=voxel_synapse_count_path)

    chunks = [TestSampleChunk(chunk_num=i)._get_xyzs_count() for i in range(n_chunks)]
    res = np.concatenate(chunks)

    # Counts should be 0, 1, 2, 3, ...
    counts = res[:, -1]
    npt.assert_array_equal(counts, np.arange(len(res)))

    len_chunks = [*map(len, chunks)]
    npt.assert_array_equal(len_chunks, [201, 201, 201, 201, 196])

    # check voxels' dimensions and coordinates
    voxel_dims = res[:, 3:6] - res[:, :3]  # max_xyz - min_xyz
    npt.assert_array_equal(voxel_dims, np.tile(voxel_counts.voxel_dimensions, (len(res), 1)))

    indices = voxel_counts.positions_to_indices(res[:, :3])
    npt.assert_array_equal(indices, voxel_order.to_numpy())
