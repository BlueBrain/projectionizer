import os
import tempfile

import pytest
from luigi import Parameter, Task, build
from luigi.local_target import LocalTarget
from mock import Mock, patch
from numpy.testing import assert_array_equal

from projectionizer import luigi_utils as lu

from utils import setup_tempdir


def test_camel2spinal_case():
    assert (lu.camel2spinal_case('CamelCase') ==
        'camel-case')

def test_resolve_morphology_config():
    class Run():
        MorphologyPath = '/fake_path/'

    config = Mock(Run=Run())
    res = lu.resolve_morphology_config(config)

    assert res == ('/fake_path/ascii', 'asc')


def test_FolderTask():
    with setup_tempdir('test_luigi') as tmp_dir:
        temp_name = os.path.join(tmp_dir, next(tempfile._RandomNameSequence()))
        assert not os.path.exists(temp_name)

        task = lu.FolderTask(folder=temp_name)
        task.run()
        assert os.path.exists(temp_name)
        assert os.path.isdir(temp_name)
        assert isinstance(task.output(), LocalTarget)


def test_common_params():
    params = {'physiology_path': 'a/fake/path',
              'folder': '/none/existant/path',
              'n_total_chunks': 'n_chunks',
              'sgid_offset': 0,
              'oversampling': 0,
              'layers': [6, 5, 4, 3, 2, 1],
              'regions': ['region_1', 'region_2'],
              }

    class TestCommonParams(lu.CommonParams):
        extension = 'out'

    with setup_tempdir('test_luigi') as tmp_dir:
        task = TestCommonParams(circuit_config=os.path.join(tmp_dir, "CircuitConfig"), **params)
    assert task.output().path == '/none/existant/path/test-common-params.out'

    assert_array_equal(params['regions'], task.get_regions())
    task.regions = []

    with patch('projectionizer.luigi_utils.read_regions_from_manifest') as patched:
        patched.return_value = []
        pytest.raises(AssertionError, task.get_regions)

    path = './relative_path'
    assert task.load_data(path) is path
    path = 'file_name.txt'
    assert task.load_data(path) is not path

    class TestCommonParamsChunk(TestCommonParams):
        chunk_num = 42

    with setup_tempdir('test_luigi') as tmp_dir:
        chunked_task = TestCommonParamsChunk(
            circuit_config=os.path.join(tmp_dir, "CircuitConfig"), **params)
    assert chunked_task.output().path == '/none/existant/path/test-common-params-chunk-42.out'

    assert isinstance(task.requires(), lu.FolderTask)


def test_RunAnywayTargetTempDir():
    with setup_tempdir('test_luigi') as tmp_dir:
        path = os.path.join(tmp_dir, 'luigi-tmp')  # directory created by RunAnywayTargetTempDir
        assert not os.path.exists(path)

        class Test(Task):
            def run(self):
                with open(self.output().path, 'w', encoding='utf-8') as fd:
                    fd.write('test')
            def output(self):
                return LocalTarget(os.path.join(tmp_dir, 'out.txt'))

        class DoAll(Task):
            """Launch the full projectionizer pipeline"""
            folder = Parameter()
            def requires(self):
                return Test()
            def run(self):
                self.output().done()
            def output(self):
                return lu.RunAnywayTargetTempDir(self, base_dir=self.folder)

        build([DoAll(folder=tmp_dir)], local_scheduler=True)
        assert os.path.exists(path)
