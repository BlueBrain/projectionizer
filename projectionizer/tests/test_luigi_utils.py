import os
import tempfile
import pytest

from luigi import Task, build, Parameter
from luigi.parameter import ParameterException
from luigi.local_target import LocalTarget
from projectionizer import luigi_utils as lu
from utils import setup_tempdir


def test_camel2spinal_case():
    assert (lu.camel2spinal_case('CamelCase') ==
        'camel-case')


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
    params = {'circuit_config': 'circuit',
              'morphology_path': 'a/fake/path',
              'recipe_path': 'a/fake/path',
              'folder': '/none/existant/path',
              'n_total_chunks': 'n_chunks',
              'sgid_offset': 0,
              'oversampling': 0,
              'layers': [(6, 700.37845971),
                         (5, 525.05585701),
                         (4, 189.57183895),
                         (3, 352.92508322),
                         (2, 148.87602025),
                         (1, 164.94915873), ],
              }

    class TestCommonParams(lu.CommonParams):
        extension = 'out'

    task = TestCommonParams(**params)
    assert task.output().path == '/none/existant/path/test-common-params.out'

    class TestCommonParamsChunk(TestCommonParams):
        chunk_num = 42

    chunked_task = TestCommonParamsChunk(**params)
    assert chunked_task.output().path == '/none/existant/path/test-common-params-chunk-42.out'

    assert isinstance(task.requires(), lu.FolderTask)

    # Test deprecation
    deprecated = params.copy()
    deprecated['voxel_path'] = '/fake/path'
    pytest.raises(ParameterException, TestCommonParams, **deprecated)

    deprecated = params.copy()
    deprecated['hex_fiber_locations'] = '/fake/path'
    pytest.raises(ParameterException, TestCommonParams, **deprecated)


def test_RunAnywayTargetTempDir():
    with setup_tempdir('test_luigi') as tmp_dir:
        path = os.path.join(tmp_dir, 'luigi-tmp')  # directory created by RunAnywayTargetTempDir
        assert not os.path.exists(path)

        class Test(Task):
            def run(self):
                with open(self.output().path, 'w') as fd:
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
