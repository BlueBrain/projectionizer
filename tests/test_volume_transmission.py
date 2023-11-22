from unittest.mock import Mock, patch

import h5py
import luigi
import numpy as np
import pandas as pd
import pytest
from luigi import Task
from numpy.testing import assert_array_equal

import projectionizer.volume_transmission as test_module
from projectionizer import step_3_write

from utils import as_iterable


@pytest.mark.MockTask(cls=test_module.VolumeRunAll)
def test_parameter_sharing(MockTask):
    """Test that the shared SONATA parameters are correctly shared and passed forward.

    I.e., check that shared parameter values are actually shared and those that should differ,
    actually differ between different projectionizer tasks.
    """
    original_config = {
        "mtype": "test_mtype",
        "node_file_name": "fake_nodes.h5",
        "edge_file_name": "fake_edges.h5",
        "node_population": "test_node_pop",
        "edge_population": "test_edge_pop",
    }

    for param, value in original_config.items():
        setattr(MockTask, param, value)

    def _check_params(task, expected_config):
        if isinstance(task, test_module.VolumeCheckSonataOutput):
            # `edge_population`, `edge_file_name` hard-coded for the task (and its sub-tasks)
            expected_config = {
                **expected_config,
                "edge_population": test_module.VolumeCheckSonataOutput.edge_population,
                "edge_file_name": test_module.VolumeCheckSonataOutput.edge_file_name,
            }
        elif isinstance(task, test_module.VolumeRunParquetConverter):
            # `edge_file_name` hard-coded for the task (and its sub-tasks)
            expected_config = {
                **expected_config,
                "edge_file_name": test_module.VolumeRunParquetConverter.edge_file_name,
            }
        elif isinstance(task, step_3_write.WriteSonataEdges) and not isinstance(
            task, test_module.VolumeWriteSonataEdges
        ):
            # superfluous sanity check; step_3_write tasks not affected by `VT` values
            assert task.edge_population == original_config["edge_population"]
            assert task.edge_file_name == original_config["edge_file_name"]

        for param, expected_value in expected_config.items():
            if hasattr(task, param):
                assert getattr(task, param) == expected_value

        try:
            for subtask in as_iterable(task.requires()):
                _check_params(subtask, expected_config)
        except luigi.parameter.MissingParameterException:
            # At this point we are wandering off from SONATA related tasks
            pass

    _check_params(MockTask(), original_config)

    # If MockVolumeCheckSonataOutput step_3_write.WriteSonataEdges, above should fail:
    class MockVolumeCheckSonataOutput(test_module.VolumeCheckSonataOutput):
        def requires(self):
            return self.clone(step_3_write.WriteSonataEdges)

    with patch.object(test_module, "VolumeCheckSonataOutput", MockVolumeCheckSonataOutput):
        expected = original_config["edge_population"]
        actual = test_module.VolumeCheckSonataOutput.edge_population

        with pytest.raises(AssertionError, match=f"assert '{actual}' == '{expected}'"):
            _check_params(MockTask(), original_config)


@patch.object(test_module, "map_parallelize", new=lambda *args, **_: map(*args))
@patch.object(test_module.spatial_index, "open_index", new=Mock())
@patch.object(test_module, "spherical_sampling")
def test_get_spherical_samples(mock_sample):
    fill_value = 5
    mock_sample.return_value = pd.DataFrame(np.full((3, 5), fill_value))
    syns = pd.DataFrame(np.ones((10, 4)), columns=list("xyz") + ["sgid"])
    ret = test_module._get_spherical_samples(syns, "fake", 1)
    assert np.all(ret.to_numpy() == fill_value)


@pytest.mark.MockTask(cls=test_module.VolumeSample)
def test_VolumeSample(MockTask):
    class TestVolumeSample(MockTask):
        additive_path_distance = 10

        def input(self):
            return (
                Mock(path=self.folder / "fake_path"),
                Mock(path=self.folder / "another_fake_path"),
            )

    samples = pd.DataFrame(
        np.ones((2, 7)), columns=list("xyz") + ["gid", "sgid", "section_id", "segment_id"]
    )
    samples.sgid = [0, 1]
    fibers = pd.DataFrame(np.ones((2, 6)), columns=list("xyzuvw"))
    distances = np.array([100, 100])

    @patch.object(test_module, "load", return_value=fibers)
    @patch.object(test_module, "_get_spherical_samples", return_value=samples)
    @patch.object(test_module, "calc_pathlength_to_fiber_start", return_value=distances)
    def run_tests(*_):
        test = TestVolumeSample()
        test.run()

        filepath = test.folder / test.output().path
        assert filepath.is_file()

        df = pd.read_feather(filepath)
        # check that all the columns are translated to snake_case
        assert all(c in df.columns for c in ("section_id", "segment_id", "tgid"))

        assert_array_equal(df["sgid_path_distance"], distances + test.additive_path_distance)

    run_tests()


@patch.object(test_module, "load", new=Mock())
@patch.object(test_module, "calculate_conductance_scaling_factor", return_value=[0])
@pytest.mark.MockTask(cls=test_module.ScaleConductance)
def test_ScaleConductance(mock_calculate, MockTask):
    class TestScaleConductance(MockTask):
        def input(self):
            return (
                Mock(path=self.folder / "fake.ext"),
                Mock(path=self.folder / "fake_syns"),
            )

        def run(self):
            edge_population = self.requires()[0].edge_population

            with h5py.File(self.input()[0].path, "w") as h5:
                population_path = f"/edges/{edge_population}/0"
                group = h5.create_group(population_path)
                group["distance_volume_transmission"] = [0]
                group["conductance"] = [0]
            super().run()

    test = TestScaleConductance()
    assert len(test.requires()) == 2
    assert all(isinstance(t, Task) for t in test.requires())

    test.run()
    assert (test.folder / test.output().path).is_file()

    # Test that the file is removed on error
    mock_calculate.side_effect = RuntimeError("fake error")
    with pytest.raises(RuntimeError):
        test.run()
    assert not (test.folder / test.output().path).exists()
