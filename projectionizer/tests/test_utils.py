import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from mock import Mock, patch
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas

import projectionizer.utils as test_module

from utils import TEST_DATA_DIR, setup_tempdir


def test_choice():
    np.random.seed(0)
    indices = test_module.choice(np.array([[1., 2, 3, 4],
                                           [0, 0, 1, 0],
                                           [6, 5, 4, 0]]))
    assert_array_equal(indices, [2, 2, 1])


def test_ignore_exception():
    with test_module.ignore_exception(OSError):
        raise OSError('This should not be propagated')

    def raise_error():
        with test_module.ignore_exception(OSError):
            raise KeyError('This should be propagated')

    pytest.raises(KeyError, raise_error)


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
    assert_array_equal(p, ret)


def test_normalize_probability_raises():
    p = np.array([1e-10, -2e-12])
    pytest.raises(test_module.ErrorCloseToZero, test_module.normalize_probability, p)


def test_load():
    with setup_tempdir('test_utils') as path:
        extensions = ['nrrd', 'json', 'feather', 'csv', 'yaml']
        files = {ext: os.path.join(path, 'test_load.{}'.format(ext)) for ext in extensions}
        dataframe = pd.DataFrame({'a': [1, 2, 3, 4]})
        dataframe.index.name = 'index_name'
        voxcell_obj = VoxelData(np.array([[[1, 1, 1]]]), (1, 1, 1))
        json_obj = {'a': 1}
        yaml_obj = json_obj

        test_module.write_feather(files['feather'], dataframe)
        dataframe.to_csv(files['csv'])
        voxcell_obj.save_nrrd(files['nrrd'])
        with open(files['json'], 'w', encoding='utf-8') as outputf:
            json.dump(json_obj, outputf)
        with open(files['yaml'], 'w', encoding='utf-8') as outputf:
            yaml.dump(yaml_obj, outputf)

        for ext, result in zip(extensions, [VoxelData, dict, pd.DataFrame, pd.DataFrame]):
            assert isinstance(test_module.load(files[ext]), result)

        class Task:
            def __init__(self, _path):
                self.path = _path

        nrrd, _json, feather, csv, _yaml = test_module.load_all(
            [Task(files[ext]) for ext in extensions]
        )

        assert dataframe.equals(feather)
        assert dataframe.equals(csv)
        assert_array_equal(voxcell_obj.raw, nrrd.raw)
        assert json_obj == _json
        assert yaml_obj == _yaml


def test_load_raise():
    pytest.raises(NotImplementedError, test_module.load, 'file.blabla')


def times_two(x):
    return x * 2


def test_map_parallelize():
    os.environ['PARALLEL_VERBOSE'] = 'True'
    mock_util = Mock()
    with patch('projectionizer.utils.multiprocessing.util.log_to_stderr', mock_util):
        a = np.arange(10)
        assert_array_equal(test_module.map_parallelize(times_two, a), a * 2)
        mock_util.assert_called()


def test_min_max_axis():
    min_xyz = np.array([0, 0, 0])
    max_xyz = np.array([1, 1, 1])
    min_, max_ = test_module.min_max_axis(min_xyz, max_xyz)
    assert_allclose(min_xyz, min_)
    assert_allclose(max_xyz, max_)

    min_xyz = np.array([-10, -5, 1])
    max_xyz = np.array([-1, 0, -1])
    min_, max_ = test_module.min_max_axis(min_xyz, max_xyz)
    assert_allclose(min_, np.array([-10, -5, -1]))
    assert_allclose(max_, np.array([-1, 0, 1]))


def test_in_bounding_box():
    in_bounding_box = test_module.in_bounding_box
    min_xyz = np.array([0, 0, 0], dtype=float)
    max_xyz = np.array([10, 10, 10], dtype=float)

    for axis in ('x', 'y', 'z'):
        df = pd.DataFrame({'x': np.arange(1, 2),
                           'y': np.arange(1, 2),
                           'z': np.arange(1, 2)})
        ret = in_bounding_box(min_xyz, max_xyz, df)
        assert_array_equal(ret.values, [True, ])

        # check for violation of min_xyz
        df[axis].iloc[0] = 0
        ret = in_bounding_box(min_xyz, max_xyz, df)
        assert_array_equal(ret.values, [False, ])
        df[axis].iloc[0] = 1

        # check for violation of max_xyz
        df[axis].iloc[0] = 10
        ret = in_bounding_box(min_xyz, max_xyz, df)
        assert_array_equal(ret.values, [False, ])
        df[axis].iloc[0] = 1

    df = pd.DataFrame({'x': np.arange(0, 10),
                       'y': np.arange(5, 15),
                       'z': np.arange(5, 15)})
    ret = in_bounding_box(min_xyz, max_xyz, df)
    assert_array_equal(ret.values, [False,  # x == 0, fails
                                    True, True, True, True,
                                    False, False, False, False, False,  # y/z > 9
                                    ])


def test_calculate_synapse_conductance():

    radius = 5
    interval = np.array((1, .1))
    distance = np.array((0, radius / 2, radius, radius + 1))
    conductance = np.ones(len(distance))

    res = test_module.calculate_synapse_conductance(
        conductance, distance, max_radius=radius, interval=interval)
    expected = np.array([interval[0], interval.mean(), interval[1], 0])

    assert_array_almost_equal(res, expected)

    conductance = np.random.random(len(distance))
    res = test_module.calculate_synapse_conductance(
        conductance, distance, max_radius=radius, interval=interval)
    assert_array_almost_equal(res, conductance * expected)


def test_mask_by_region():
    atlas = Atlas.open(TEST_DATA_DIR)
    mask = test_module.mask_by_region(['TEST_layers'], atlas)
    assert mask.sum() == 60 * 28 * 28

    mask = test_module.mask_by_region([10], atlas)
    assert mask.sum() == 60 * 28 * 28

    assert pytest.raises(KeyError, test_module.mask_by_region, ['Fake_layers'], atlas)


def test_regex_to_regions():
    reg_str = '@^region_1$'
    res = test_module._regex_to_regions(reg_str)

    assert_array_equal(res, ['region_1'])

    reg_str = r'@^(region_1\|region_2)$'
    res = test_module._regex_to_regions(reg_str)

    assert_array_equal(res, ['region_1', 'region_2'])


def test_read_regions_from_manifest():
    with setup_tempdir('test_utils') as dirpath:
        circuit_config = os.path.join(dirpath, 'CircuitConfig')
        manifest_path = os.path.join(dirpath, 'MANIFEST.yaml')
        manifest_obj = {'common': {'region': 'R1'}}

        with open(circuit_config, 'w', encoding='utf-8') as fd:
            fd.write('')
        with open(manifest_path, 'w', encoding='utf-8') as fd:
            yaml.dump(manifest_obj, fd)

        mock_config = Mock(return_value=Mock(Run=Mock(BioName=dirpath)))
        with patch('projectionizer.utils.BlueConfig', mock_config):
            res = test_module.read_regions_from_manifest(circuit_config)
            assert_array_equal(res, [manifest_obj['common']['region']])

            with open(manifest_path, 'w', encoding='utf-8') as fd:
                yaml.dump({}, fd)

            res = test_module.read_regions_from_manifest(circuit_config)
            assert_array_equal(res, [])


def test_convert_to_smallest_allowed_int_type():

    res = test_module.convert_to_smallest_allowed_int_type(np.array([0, 1]))
    assert_array_equal(res, [0, 1])
    assert res.dtype == np.int16

    res = test_module.convert_to_smallest_allowed_int_type(np.array([0, int(2**17)]))
    assert res.dtype == np.int32

    res = test_module.convert_to_smallest_allowed_int_type(np.array([0, int(2**33)]))
    assert res.dtype == np.int64


def test_convert_layer_to_PH_format():
    layers = [f'L{n}' for n in range(11)] + ['layer_name', 'Layer_1', 'L23', 'L3a']
    expected = [f'{n}' for n in range(10)] + ['L10', 'layer_name', 'Layer_1', 'L23', 'L3a']
    ret = [test_module.convert_layer_to_PH_format(l) for l in layers]
    assert_array_equal(ret, expected)


def test_delete_file_on_exception():
    with setup_tempdir('test_utils') as test_dir:
        # test that file exists if no exceptions
        test_file = os.path.join(test_dir, 'test.txt')
        with test_module.delete_file_on_exception(test_file):
            with open(test_file, 'w', encoding='utf-8') as fd:
                fd.write('')

        assert os.path.exists(test_file)

        # test file removal on exception
        try:
            with test_module.delete_file_on_exception(test_file):
                assert os.path.exists(test_file)
                raise IOError('')
        except IOError:
            pass

        assert not os.path.exists(test_file)
