import json
import os

import numpy as np
import pandas as pd
from nose.tools import eq_, ok_, raises
from numpy.testing import assert_equal
from voxcell import VoxelData

from projectionizer import utils
from utils import setup_tempdir


def test_write_feather():
    with setup_tempdir('test_utils') as path:
        path = os.path.join(path, 'projectionizer_test_write.feather')
        data = pd.DataFrame({'a': [1, 2, 3, 4]})
        utils.write_feather(path, data)
        new_data = utils.load(path)
        pd.testing.assert_frame_equal(data, new_data)


def test_normalize_probability():
    p = np.array([1, 0])
    ret = utils.normalize_probability(p)
    assert_equal(p, ret)


@raises(utils.ErrorCloseToZero)
def test_normalize_probability_raises():
    p = np.array([1e-10, -2e-12])
    utils.normalize_probability(p)


def test_load():
    with setup_tempdir('test_utils') as path:
        feather_file = os.path.join(path, 'test_load.feather')
        nrrd_file = os.path.join(path, 'test_load.nrrd')
        json_file = os.path.join(path, 'test_load.json')
        feather_obj = pd.DataFrame({'a': [1, 2, 3, 4]})
        voxcell_obj = VoxelData(np.array([[[1, 1, 1]]]), (1, 1, 1))
        json_obj = {'a': 1}
        utils.write_feather(feather_file, feather_obj)
        voxcell_obj.save_nrrd(nrrd_file)
        with open(json_file, 'w') as outputf:
            json.dump(json_obj, outputf)

        ok_(isinstance(utils.load(feather_file), pd.DataFrame))
        ok_(isinstance(utils.load(nrrd_file), VoxelData))
        ok_(isinstance(utils.load(json_file), dict))

        class Task(object):
            def __init__(self, _path):
                self.path = _path

        f, v, j = utils.load_all([Task(feather_file), Task(nrrd_file), Task(json_file)])
        ok_(feather_obj.equals(f))
        assert_equal(voxcell_obj.raw, v.raw)
        eq_(json_obj, j)


@raises(NotImplementedError)
def test_load_raise():
    utils.load('file.blabla')


def times_two(x):
    return x * 2


def test_map_parallelize():
    a = np.arange(10)
    assert_equal(utils.map_parallelize(times_two, a),
                 a * 2)


def test_in_bounding_box():
    in_bounding_box = utils.in_bounding_box
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
