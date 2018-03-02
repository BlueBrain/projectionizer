from nose.tools import eq_

from projectionizer import luigi_utils as lu


def test_camel2spinal_case():
    eq_(lu.camel2spinal_case('CamelCase'),
        'camel-case')


def test_cloned_tasks():
    class Class:
        def clone(self, x):
            return x
    eq_(lu.cloned_tasks(Class(), [1, 2, 3]),
        [1, 2, 3])


def test_common_params():
    class BlaBlaTask(lu.CommonParams):
        circuit_config = 'circuit'
        folder = '/none/existant/path'
        extension = 'ext'
        geometry = 'geo'
        n_total_chunks = 'n_chunks'
        sgid_offset = 0
        oversampling = 0
        voxel_path = ''
        prefix = ''
        extension = 'out'

    task = BlaBlaTask()
    eq_(task.output().path, '/none/existant/path/bla-bla-task.out')

    class BlaBlaChunk(BlaBlaTask):
        chunk_num = 42

    chunked_task = BlaBlaChunk()
    eq_(chunked_task.output().path, '/none/existant/path/bla-bla-chunk-42.out')
