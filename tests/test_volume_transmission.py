import os

import h5py
import numpy as np
import pandas as pd
from bluepy import Section, Segment
from luigi import Task
from mock import Mock, patch
from numpy.testing import assert_array_equal, assert_equal, assert_raises

from projectionizer import volume_transmission as test_module

from utils import setup_tempdir


def test_get_spherical_samples():
    syns = pd.DataFrame(np.ones((10, 4)), columns=list("xyz") + ["sgid"])
    with patch("projectionizer.volume_transmission.map_parallelize") as patched:
        patched.return_value = [
            pd.DataFrame(np.ones((3, 5))) * 5,
            pd.DataFrame(np.ones((3, 5))) * 5,
        ]

        ret = test_module._get_spherical_samples(syns, "fake", 1)
        assert np.all(ret.to_numpy() == 5)


def test_VolumeSample():
    samples = pd.DataFrame(
        np.ones((2, 7)), columns=list("xyz") + ["gid", "sgid", Section.ID, Segment.ID]
    )
    samples.sgid = [0, 1]
    fibers = pd.DataFrame(np.ones((2, 6)), columns=list("xyzuvw"))
    distances = np.full(2, 100)

    with setup_tempdir("test_volume_sample") as tmp_folder:

        class TestVolumeSample(test_module.VolumeSample):
            folder = tmp_folder
            physiology_path = "fake_string"
            circuit_config = os.path.join(tmp_folder, "CircuitConfig")
            sgid_offset = n_total_chunks = oversampling = None
            layers = ""
            additive_path_distance = radius = 10

            def input(self):
                return (
                    Mock(path=os.path.join(tmp_folder, "fake_path")),
                    Mock(path=os.path.join(tmp_folder, "another_fake_path")),
                )

        @patch.object(test_module, "load", return_value=fibers)
        @patch.object(test_module, "_get_spherical_samples", return_value=samples)
        @patch.object(test_module, "calc_pathlength_to_fiber_start", return_value=distances)
        def run_tests(*_):
            test = TestVolumeSample()
            test.run()

            filepath = os.path.join(tmp_folder, test.output().path)
            assert os.path.isfile(filepath)

            df = pd.read_feather(filepath)
            assert_equal({"section_id", "segment_id", "tgid"} - {*df.columns}, set())

            assert_array_equal(df["sgid_path_distance"], np.full(2, 110))

        run_tests()


def test_ScaleConductance():
    with setup_tempdir("test_scale_conductance") as tmp_folder:

        class TestScaleConductance(test_module.ScaleConductance):
            folder = tmp_folder
            physiology_path = "fake_string"
            circuit_config = os.path.join(tmp_folder, "CircuitConfig")
            sgid_offset = n_total_chunks = oversampling = None
            edge_population = "fake_edges"
            layers = ""

            def input(self):
                return (
                    Mock(path=os.path.join(tmp_folder, "fake.ext")),
                    Mock(path=os.path.join(tmp_folder, "fake_syns")),
                )

            def run(self):
                edge_population = self.requires()[0].edge_population

                with h5py.File(self.input()[0].path, "w") as h5:
                    population_path = f"/edges/{edge_population}/0"
                    group = h5.create_group(population_path)
                    group["distance_volume_transmission"] = [0]
                    group["conductance"] = [0]
                super().run()

        with patch(f"{test_module.__name__}.calculate_conductance_scaling_factor") as patched:
            patched.return_value = [0]

            with patch("projectionizer.volume_transmission.load"):
                test = TestScaleConductance()
                assert len(test.requires()) == 2
                assert all(isinstance(t, Task) for t in test.requires())

                test.run()
                assert os.path.isfile(os.path.join(tmp_folder, test.output().path))

                # Test that the file is removed on error
                patched.side_effect = RuntimeError("fake error")
                assert_raises(RuntimeError, test.run)
                assert not os.path.exists(os.path.join(tmp_folder, test.output().path))
