import os
import tempfile
from nose.tools import eq_, ok_

from luigi import Task
from luigi.local_target import LocalTarget
from projectionizer import luigi_utils as lu
from utils import setup_tempdir


def test_cloned_tasks():
    class Class(object):
        def clone(self, x):
            return x
    eq_(lu.cloned_tasks(Class(), [1, 2, 3]),
        [1, 2, 3])


def test_camel2spinal_case():
    eq_(lu.camel2spinal_case('CamelCase'),
        'camel-case')


def test_FolderTask():
    with setup_tempdir('test_luigi') as tmp_dir:
        temp_name = os.path.join(tmp_dir, tempfile._RandomNameSequence().next())
        ok_(not os.path.exists(temp_name))

        task = lu.FolderTask(folder=temp_name)
        task.run()
        ok_(os.path.exists(temp_name))
        ok_(os.path.isdir(temp_name))
        ok_(isinstance(task.output(), LocalTarget))


def test_common_params():
    params = {'circuit_config': 'circuit',
              'folder': '/none/existant/path',
              'geometry': 'geo',
              'n_total_chunks': 'n_chunks',
              'sgid_offset': 0,
              'oversampling': 0,
              'voxel_path': '',
              'prefix': '',
              }

    class TestCommonParams(lu.CommonParams):
        extension = 'out'

    task = TestCommonParams(**params)
    eq_(task.output().path, '/none/existant/path/test-common-params.out')

    class TestCommonParamsChunk(TestCommonParams):
        chunk_num = 42

    chunked_task = TestCommonParamsChunk(**params)
    eq_(chunked_task.output().path, '/none/existant/path/test-common-params-chunk-42.out')

    ok_(isinstance(task.requires(), lu.FolderTask))


def test_RunAnywayTargetTempDir():
    class Test(Task):
        pass

    with setup_tempdir('test_luigi') as tmp_dir:
        lu.RunAnywayTargetTempDir(Test(), tmp_dir)
        eq_(len(os.listdir(tmp_dir)), 1)  # directory created by RunAnywayTargetTempDir
