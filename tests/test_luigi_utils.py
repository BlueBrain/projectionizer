import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from luigi import PathParameter, Task, build
from luigi.local_target import LocalTarget
from numpy.testing import assert_array_equal

import projectionizer.luigi_utils as test_module
from projectionizer.version import VERSION


@pytest.mark.parametrize(
    "archive,raises",
    [
        ("unstable", False),
        ("archive/2024-01", False),
        ("archive/2023-06", False),
        ("random", True),
        ("archive/2021-12", True),
        ("archive/2023-05", True),
        ("arhive/2024-01", True),
        ("arc-hive/2024-01", True),
        ("archive/24-01", True),
        ("archive/2024-1", True),
        ("archive/2024-012", True),
    ],
)
def test__check_module_archive(archive, raises):
    if raises:
        with pytest.raises(ValueError, match=re.escape(f"Invalid module archive: '{archive}'.")):
            test_module._check_module_archive(archive)
    else:
        assert test_module._check_module_archive(archive) is None


def test__check_version_compatibility():
    assert test_module._check_version_compatibility(VERSION) is None

    with pytest.raises(
        RuntimeError, match="Given config file is intended for projectionizer version '0.0.1'."
    ):
        assert test_module._check_version_compatibility("0.0.1")

    with pytest.raises(
        ValueError, match="Expected projectionizer version to be given in format 'X.Y.Z'"
    ):
        assert test_module._check_version_compatibility("0.1")


def test_camel2spinal_case():
    assert test_module.camel2spinal_case("CamelCase") == "camel-case"


def test_resolve_morphology_config():
    class Run:
        # has no MorphologyType, expect to fall back to ASCII
        MorphologyPath = "/fake_path/"

    config = Mock(Run=Run())
    res = test_module.resolve_morphology_config(config)

    assert res == (Path("/fake_path/ascii"), "asc")


def test_FolderTask(tmp_confdir):
    temp_name = tmp_confdir / "test_folder"
    assert not temp_name.exists()

    task = test_module.FolderTask(folder=temp_name)
    task.run()
    assert temp_name.exists()
    assert temp_name.is_dir()
    assert isinstance(task.output(), LocalTarget)


@patch.object(test_module, "read_regions_from_manifest", new=Mock(return_value=[]))
@pytest.mark.MockTask(cls=test_module.CommonParams)
def test_common_params(MockTask):
    target_regions = ["region_1", "region_2"]

    class TestCommonParams(MockTask):
        extension = "out"
        regions = target_regions

    task = TestCommonParams()

    assert task.output().path == str(task.folder / "test-common-params.out")
    assert_array_equal(target_regions, task.get_regions())

    assert isinstance(task.requires(), test_module.FolderTask)

    # Should raise when no regions are found
    task.regions = []
    pytest.raises(AssertionError, task.get_regions)

    path = "./relative_path"
    assert task.load_data(path) is path

    # returns path to templates when no '/' in path
    path = "file_name.txt"
    assert task.load_data(path) == test_module.TEMPLATES_PATH / path


@pytest.mark.MockTask(cls=test_module.CommonParams)
def test_common_params_chunk(MockTask):
    class TestCommonParamsChunk(MockTask):
        extension = "out"
        chunk_num = 42

    chunked_task = TestCommonParamsChunk()

    assert chunked_task.output().path == str(
        chunked_task.folder / "test-common-params-chunk-42.out"
    )

    assert isinstance(chunked_task.requires(), test_module.FolderTask)


def test_RunAnywayTargetTempDir(tmp_confdir):
    path = tmp_confdir / "luigi-tmp"  # directory created by RunAnywayTargetTempDir
    assert not path.exists()

    class Test(Task):
        def run(self):
            with open(self.output().path, "w", encoding="utf-8") as fd:
                fd.write("test")

        def output(self):
            return LocalTarget(tmp_confdir / "out.txt")

    class DoAll(Task):
        """Launch the full projectionizer pipeline"""

        folder = PathParameter()

        def requires(self):
            return Test()

        def run(self):
            self.output().done()

        def output(self):
            return test_module.RunAnywayTargetTempDir(self, base_dir=self.folder)

    build([DoAll(folder=tmp_confdir)], local_scheduler=True)
    assert path.exists()
