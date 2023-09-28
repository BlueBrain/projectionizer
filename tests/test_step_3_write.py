import logging
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd
import pytest
from luigi import Parameter, Task

import projectionizer
import projectionizer.step_3_write as test_module

from utils import EDGE_POPULATION, NODE_POPULATION

logging.basicConfig()
L = logging.getLogger(__name__)


@pytest.mark.MockTask(cls=test_module.WriteSonata)
def test_WriteSonata(MockTask):
    syns_path = MockTask.folder / "mock_synapses.feather"
    df = pd.DataFrame(
        {
            "tgid": [10],
            "sgid": [20],
            "section_id": [1033],
            "segment_id": [1033],
            "section_type": [3],
            "section_pos": [0.5],
            "synapse_offset": [128.0],
            "x": [101],
            "y": [102],
            "z": [103],
            "sgid_path_distance": [0.5],
        }
    )
    projectionizer.utils.write_feather(syns_path, df)

    def create_h5_files(sonata, node, edge):
        with h5py.File(sonata, "w") as h5:
            group = h5.create_group(f"edges/{EDGE_POPULATION}")
            group["source_node_id"] = [0] * len(df.sgid)
        with h5py.File(node, "w") as h5:
            group = h5.create_group(f"nodes/{NODE_POPULATION}")
            group["node_type_id"] = np.full(df.sgid.max() + 1, -1)
        with h5py.File(edge, "w") as h5:
            group = h5.create_group(f"edges/{EDGE_POPULATION}")
            group["source_node_id"] = df.sgid.to_numpy()
            group["target_node_id"] = df.tgid.to_numpy()

    class TestWriteSonata(MockTask):
        node_population = NODE_POPULATION
        edge_population = EDGE_POPULATION
        node_file_name = "nodes.h5"
        edge_file_name = "edges.h5"

        def input(self):
            sonata_path = self._get_full_path_output(self.edge_file_name)
            node_path = self._get_full_path_output(self.node_file_name)
            edge_path = self.clone(test_module.WriteSonataEdges).output().path
            return (
                Mock(path=sonata_path),
                Mock(path=syns_path),
                Mock(path=node_path),
                Mock(path=edge_path),
            )

    test = TestWriteSonata()
    assert len(test.requires()) == 4
    assert all(isinstance(t, Task) for t in test.requires())
    assert test.requires()[0].output().path == test.output().path

    sonata_path = test.input()[0].path
    node_path = test.input()[2].path
    edge_path = test.input()[3].path

    # Clean run
    create_h5_files(sonata_path, node_path, edge_path)
    test.run()

    # Wrong edge count
    with h5py.File(sonata_path, "r+") as h5:
        del h5[f"edges/{EDGE_POPULATION}/source_node_id"]
        h5[f"edges/{EDGE_POPULATION}/source_node_id"] = [0] * (len(df.sgid) + 1)
    pytest.raises(AssertionError, test.run)

    # Wrong node count
    create_h5_files(sonata_path, node_path, edge_path)
    with h5py.File(node_path, "r+") as h5:
        del h5[f"nodes/{NODE_POPULATION}/node_type_id"]
        h5[f"nodes/{NODE_POPULATION}/node_type_id"] = np.full(df.sgid.max() + 10, -1)
    pytest.raises(AssertionError, test.run)

    # SGIDs are off
    create_h5_files(sonata_path, node_path, edge_path)
    with h5py.File(edge_path, "r+") as h5:
        del h5[f"edges/{EDGE_POPULATION}/source_node_id"]
        h5[f"edges/{EDGE_POPULATION}/source_node_id"] = df.sgid.to_numpy() - 1
    pytest.raises(AssertionError, test.run)

    # TGIDs are off
    create_h5_files(sonata_path, node_path, edge_path)
    with h5py.File(edge_path, "r+") as h5:
        del h5[f"edges/{EDGE_POPULATION}/target_node_id"]
        h5[f"edges/{EDGE_POPULATION}/target_node_id"] = df.tgid.to_numpy() - 1
    pytest.raises(AssertionError, test.run)


@pytest.mark.MockTask(cls=test_module.WriteSonataNodes)
def test_WriteSonataNodes(MockTask):
    mock_path = MockTask.folder / "mock_synapses.feather"
    data = {
        "tgid": [10],
        "sgid": [20],
        "section_id": [1033],
        "segment_id": [1033],
        "section_type": [3],
        "section_pos": [0.5],
        "synapse_offset": [128.0],
        "x": [101],
        "y": [102],
        "z": [103],
        "sgid_path_distance": [0.5],
    }
    projectionizer.utils.write_feather(mock_path, pd.DataFrame(data))

    class TestWriteSonataNodes(MockTask):
        node_population = NODE_POPULATION
        edge_population = EDGE_POPULATION
        node_file_name = "nodes.h5"

        def input(self):
            return Mock(path=mock_path)

    test = TestWriteSonataNodes()
    assert isinstance(test.requires(), Task)

    test.run()
    assert Path(test.output().path).is_file()

    with h5py.File(test.output().path, "r") as h5:
        # size should be as follows since indexing starts at 0 and no offset is removed
        expected_size = max(data["sgid"]) + 1
        assert len(h5[f"nodes/{NODE_POPULATION}/node_type_id"]) == expected_size


@pytest.mark.MockTask(cls=test_module.WriteSonataEdges)
def test_WriteSonataEdges(MockTask):
    mock_syn_path = MockTask.folder / "mock_synapses.feather"
    mock_pos_path = MockTask.folder / "mock_positions.feather"
    data = {
        "tgid": [10],
        "sgid": [20],
        "section_id": [1033],
        "segment_id": [1033],
        "section_type": [3],
        "synapse_offset": [128.0],
        "x": [101],
        "y": [102],
        "z": [103],
        "sgid_path_distance": [0.5],
    }
    projectionizer.utils.write_feather(mock_syn_path, pd.DataFrame(data))
    projectionizer.utils.write_feather(mock_pos_path, pd.DataFrame({"section_pos": [0.5]}))

    class TestWriteSonataEdges(MockTask):
        node_population = NODE_POPULATION
        edge_population = EDGE_POPULATION
        edge_file_name = "edges.h5"

        def input(self):
            return Mock(path=mock_syn_path), Mock(path=mock_pos_path)

    test = TestWriteSonataEdges()
    assert isinstance(test.requires(), tuple)
    for t in test.requires():
        assert isinstance(t, Task)

    test.run()
    assert Path(test.output().path).is_file()

    with h5py.File(test.output().path, "r") as h5:
        # source and target node ids at[0] should be same as in data as no offset is removed
        assert h5[f"edges/{EDGE_POPULATION}/source_node_id"][0] == data["sgid"][0]
        assert h5[f"edges/{EDGE_POPULATION}/target_node_id"][0] == data["tgid"][0]


def test_check_if_old_syntax():
    assert not test_module._check_if_old_syntax("fake_archive")
    assert not test_module._check_if_old_syntax("archive/2021-07")
    assert test_module._check_if_old_syntax("archive/2021-06")
    assert test_module._check_if_old_syntax("archive/2020-12")


@patch.object(test_module.subprocess, "run")
@pytest.mark.MockTask(cls=test_module.RunSpykfunc)
def test_RunSpykfunc(mock_subp_run, MockTask):
    class TestRunSpykfunc(MockTask):
        module_archive = Parameter(default="")

        def run(self):
            Path(self.output().path).mkdir(parents=True, exist_ok=True)
            super().run()

    test = TestRunSpykfunc(module_archive="unstable")
    assert "--touches" not in test._parse_command()

    test = TestRunSpykfunc(module_archive="archive/2020-12")
    assert "--touches" in test._parse_command()

    assert len(test.requires()) == 2
    assert all(isinstance(t, Task) for t in test.requires())

    test.run()
    assert (test.folder / "spykfunc").is_dir()

    # Test that the spykfunc dir is removed on error
    mock_subp_run.side_effect = subprocess.CalledProcessError(1, "fake")
    pytest.raises(subprocess.CalledProcessError, test.run)
    assert not (test.folder / "spykfunc").is_dir()


@patch.object(test_module.subprocess, "run", new=Mock())
@pytest.mark.MockTask(cls=test_module.RunParquetConverter)
def test_RunParquetConverter(MockTask):
    class TestRunParquetConverter(MockTask):
        module_archive = Parameter(default="")

        def run(self):
            with open(self.output().path, "w", encoding="utf-8") as fd:
                fd.write("")
            super().run()

    test = TestRunParquetConverter(module_archive="unstable")
    assert "--format" not in test._parse_command()

    test = TestRunParquetConverter(module_archive="archive/2020-12")
    assert "--format" in test._parse_command()

    assert len(test.requires()) == 2
    assert all(isinstance(t, Task) for t in test.requires())

    test.run()
    assert (test.folder / test.edge_file_name).is_file()

    test_file_name = "fake.sonata"
    test.edge_file_name = test_file_name
    test.run()
    assert (test.folder / test_file_name).is_file()
