import numpy as np
import pandas as pd
from nose.tools import eq_, ok_, raises
from numpy.testing import assert_equal
from voxcell import VoxelData

from projectionizer.utils import *
from projectionizer.utils import _camel_case_to_spinal_case, _write_feather


def test_write_feather():
    _write_feather('/tmp/projectionizer_test_write.feather',
                   pd.DataFrame({'a': [1, 2, 3, 4]}))
    ok_(True)


def test_camel_case_to_snake_case():
    assert_equal(_camel_case_to_spinal_case('CamelCase'),
                 'camel-case')


def test_normalize_probability():
    p = np.array([1, 0])
    ret = normalize_probability(p)
    assert_equal(p, ret)


@raises(ErrorCloseToZero)
def test_normalize_probability_raises():
    p = np.array([1e-10, -2e-12])
    normalize_probability(p)


def test_load():
    feather_file = '/tmp/test_load.feather'
    nrrd_file = '/tmp/test_load.nrrd'
    json_file = '/tmp/test_load.json'
    feather_obj = pd.DataFrame({'a': [1, 2, 3, 4]})
    voxcell_obj = VoxelData(np.array([[[1, 1, 1]]]), (1, 1, 1))
    json_obj = {'a': 1}
    _write_feather(feather_file,
                   feather_obj)
    voxcell_obj.save_nrrd(nrrd_file)
    with open(json_file, 'w') as outputf:
        json.dump(json_obj, outputf)

    isinstance(load(feather_file), pd.DataFrame)
    isinstance(load(nrrd_file), VoxelData)
    isinstance(load(json_file), dict)

    class Task:
        def __init__(self, _path):
            self.path = _path

    f, v, j = load_all([Task(feather_file), Task(nrrd_file), Task(json_file)])
    ok_(feather_obj.equals(f))
    assert_equal(voxcell_obj.raw, v.raw)
    eq_(json_obj, j)


@raises(NotImplementedError)
def test_load_raise():
    load('file.blabla')


def test_cloned_tasks():
    class Class:
        def clone(self, x):
            return x
    assert_equal(cloned_tasks(Class(), [1, 2, 3]),
                 [1, 2, 3])


def times_two(x):
    return x * 2


def test_map_parallelize():
    a = np.arange(10)
    assert_equal(map_parallelize(times_two, a),
                 a * 2)


def test_in_bounding_box():
    min_xyz = np.array([0, 0, 0], dtype=float)
    max_xyz = np.array([10, 10, 10], dtype=float)

    for axis in ('x', 'y', 'z'):
        df = pd.DataFrame({'x': np.arange(1, 2),
                           'y': np.arange(1, 2),
                           'z': np.arange(1, 2)})
        ret = in_bounding_box(min_xyz, max_xyz, df)
        assert_equal(ret.values, [True, ])

        # check for violation of min_xyz
        df[axis].iloc[0] = 0
        ret = in_bounding_box(min_xyz, max_xyz, df)
        assert_equal(ret.values, [False, ])
        df[axis].iloc[0] = 1

        # check for violation of max_xyz
        df[axis].iloc[0] = 10
        ret = in_bounding_box(min_xyz, max_xyz, df)
        assert_equal(ret.values, [False, ])
        df[axis].iloc[0] = 1

    df = pd.DataFrame({'x': np.arange(0, 10),
                       'y': np.arange(5, 15),
                       'z': np.arange(5, 15)})
    ret = in_bounding_box(min_xyz, max_xyz, df)
    assert_equal(ret.values, [False,  # x == 0, fails
                              True, True, True, True,
                              False, False, False, False, False,  # y/z > 9
                              ])


def test_common_params():
    class BlaBlaTask(CommonParams):
        circuit_config = 'circuit'
        folder = '/tmp'
        extension = 'ext'
        geometry = 'geo'
        n_total_chunks = 'n_chunks'
        sgid_offset = 0
        oversampling = 0
        voxel_path = ''
        prefix = ''
        extension = 'out'

    task = BlaBlaTask()
    assert_equal(task.output().path, '/tmp/bla-bla-task.out')

    class BlaBlaChunk(BlaBlaTask):
        chunk_num = 42

    chunked_task = BlaBlaChunk()
    assert_equal(chunked_task.output().path, '/tmp/bla-bla-chunk-42.out')
