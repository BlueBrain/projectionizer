import json
import os

import numpy as np
import pandas as pd
from nose.tools import eq_, ok_, raises, assert_raises
from numpy.testing import assert_equal
from voxcell import VoxelData

import projectionizer.utils as test_module

from utils import setup_tempdir
from utils import TEST_DATA_DIR


def test_ignore_exception():
    with test_module.ignore_exception(OSError):
        raise OSError('This should not be propagated')

    def foo():
        with test_module.ignore_exception(OSError):
            raise KeyError('This should be propagated')

    assert_raises(KeyError, foo)


def test_write_feather():
    with setup_tempdir('test_utils') as path:
        path = os.path.join(path, 'projectionizer_test_write.feather')
        data = pd.DataFrame({'a': [1, 2, 3, 4]})
        test_module.write_feather(path, data)
        new_data = test_module.load(path)
        pd.testing.assert_frame_equal(data, new_data)


def test_normalize_probability():
    p = np.array([1, 0])
    ret = test_module.normalize_probability(p)
    assert_equal(p, ret)


@raises(test_module.ErrorCloseToZero)
def test_normalize_probability_raises():
    p = np.array([1e-10, -2e-12])
    test_module.normalize_probability(p)


def test_load():
    with setup_tempdir('test_utils') as path:
        extensions = ['nrrd', 'json', 'feather', 'csv']
        files = {ext: os.path.join(path, 'test_load.{}'.format(ext)) for ext in extensions}
        dataframe = pd.DataFrame({'a': [1, 2, 3, 4]})
        dataframe.index.name = 'index_name'
        voxcell_obj = VoxelData(np.array([[[1, 1, 1]]]), (1, 1, 1))
        json_obj = {'a': 1}

        test_module.write_feather(files['feather'], dataframe)
        dataframe.to_csv(files['csv'])
        voxcell_obj.save_nrrd(files['nrrd'])
        with open(files['json'], 'w') as outputf:
            json.dump(json_obj, outputf)

        for ext, result in zip(extensions, [VoxelData, dict, pd.DataFrame, pd.DataFrame]):
            ok_(isinstance(test_module.load(files[ext]), result))

        class Task(object):
            def __init__(self, _path):
                self.path = _path

        nrrd, _json, feather, csv = test_module.load_all([Task(files[ext]) for ext in extensions])

        ok_(dataframe.equals(feather))
        ok_(dataframe.equals(csv))
        assert_equal(voxcell_obj.raw, nrrd.raw)
        eq_(json_obj, _json)


@raises(NotImplementedError)
def test_load_raise():
    test_module.load('file.blabla')


def times_two(x):
    return x * 2


def test_map_parallelize():
    a = np.arange(10)
    assert_equal(test_module.map_parallelize(times_two, a),
                 a * 2)


def test_in_bounding_box():
    in_bounding_box = test_module.in_bounding_box
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


def test_mask_by_region():
    region = 'primary somatosensory cortex, hindlimb region'
    mask = test_module.mask_by_region(region, TEST_DATA_DIR, '')
    assert_equal(mask.sum(), 101857)

    mask = test_module.mask_by_region([726], TEST_DATA_DIR, '')
    assert_equal(mask.sum(), 101857)
