from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd
import pytest
from bluepy import Section, Segment
from luigi import Task
from numpy.testing import assert_array_equal

import projectionizer.volume_transmission as test_module


@patch.object(test_module, "map_parallelize")
def test_get_spherical_samples(mock_map_parallelize):
    fill_value = 5
    mock_map_parallelize.return_value = 2 * [pd.DataFrame(np.full((3, 5), fill_value))]
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
        np.ones((2, 7)), columns=list("xyz") + ["gid", "sgid", Section.ID, Segment.ID]
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
