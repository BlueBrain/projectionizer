import numpy.testing as npt
from nose import tools as nt

from examples.layer_data import *


def test_layer_spatial_data():
    nt.assert_dict_equal(layer_spatial_data([1, 1, 2]),
                         {3: {'start': 0,
                              'end': 2,
                              'thickness': 2},
                          2: {'start': 2,
                              'end': 3,
                              'thickness': 1},
                          1: {'start': 3,
                              'end': 4,
                              'thickness': 1}})


def test_density_absolute_coord():
    data = layer_spatial_data([1, 1, 2])
    density_per_fraction = ((0.2, 0.1),
                            (0.4, 0.2),
                            (0.6, 0.3),
                            (1.0, 0.5),)
    data = density_absolute_coord(
        data[3], data[2], 0, 0.6, density_per_fraction)
    npt.assert_array_equal(data, [[0.52, 0.1], [1.04, 0.2], [1.56, 0.3], [2.6, 0.5]])


def test_all_data():
    thickness = [0.2, 0.2, 1, 1, 2, 0.5]
    density_per_fraction = {4: ((0.2, 0.1),
                                (0.4, 0.2),
                                (0.6, 0.3),
                                (1.0, 0.5),),
                            6: ((0.2, 0.1),
                                (0.4, 0.2),
                                (0.6, 0.3),
                                (1.0, 0.5),)}
    data = all_data({'thickness': thickness, 'density_per_fraction': density_per_fraction})
    result = {6: {'start': 0, 'end': 0.5, 'thickness': 0.5},
              5: {'start': 0.5, 'end': 2.5, 'thickness': 2},
              4: {'start': 2.5, 'end': 3.5, 'thickness': 1},
              3: {'start': 3.5, 'end': 4.5, 'thickness': 1},
              2: {'start': 4.5, 'end': 4.7, 'thickness': 0.2},
              1: {'start': 4.7, 'end': 4.9, 'thickness': 0.2}}

    npt.assert_array_almost_equal(data[4]['density'], [[2.8, 0.1],
                                                       [3.1, 0.2],
                                                       [3.4, 0.3],
                                                       [4., 0.5]])
    npt.assert_array_almost_equal(data[6]['density'], [[0.68, 0.1],
                                                       [0.935, 0.2],
                                                       [1.19, 0.3],
                                                       [1.7, 0.5]])

    for layer in result.keys():
        if 'density' in data[layer]:
            del data[layer]['density']
        nt.assert_dict_equal(data[layer], result[layer])
        nt.assert_dict_equal(data[layer], result[layer])
        nt.assert_dict_equal(data[layer], result[layer])
