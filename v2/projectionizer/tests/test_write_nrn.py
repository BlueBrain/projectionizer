import os

import h5py
import numpy as np
import pandas as pd
from nose.tools import ok_, raises
from numpy.testing import assert_allclose, assert_equal

from projectionizer import write_nrn
from projectionizer.tests.utils import setup_tempdir

SYNAPSE_PARAMS = {'id': 1,
                  'gsyn': (3, 4),
                  'Use': (5, 6),
                  'D': (7, 8),
                  'F': (9, 10),
                  'DTC': (11, 12),
                  'Ase': (13, 14)}


def test_create_synapse_data():
    np.random.seed(1337)

    df = pd.DataFrame({'tgid': [1],
                       'y': [0.33],
                       'section_id': [1033],
                       'location': [-3],
                       'segment_id': [1033],
                       'sgid_path_distance': [300.]
                       })

    assert_allclose(write_nrn.create_synapse_data(df, SYNAPSE_PARAMS, efferent=True),
                    [[1.00000000e+00,   300. / 300.,   1.03300000e+03,   1.03300000e+03,
                      -7.86074025e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                      7.75316318e+00,   1.92565485e+01,   2.51811971e+01,   9.06247588e+01,
                      1.17854003e+02,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                      0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])


def test_write_synapses():
    np.random.seed(1337)
    df = pd.DataFrame({'tgid': [1],
                       'sgid': [2],
                       'y': [0.33],
                       'section_id': [1033],
                       'location': [3],
                       'syn_ids': [12],
                       'segment_id': [1033],
                       'sgid_path_distance': [300.]
                       })

    with setup_tempdir('test_utils') as folder:
        path = os.path.join(folder, 'projectionizer_test_write_synapses.h5')
        write_nrn.write_synapses(path, df.groupby('sgid'),
                                 SYNAPSE_PARAMS, efferent=False)

        correct_result = [[2.00000000e+00, 1.00000000e+00, 1.03300000e+03, 1.03300000e+03,
                           7.86074025e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                           7.75316318e+00, 1.92565485e+01, 2.51811971e+01, 9.06247588e+01,
                           1.17854003e+02, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ]]

        with h5py.File(path) as h5:
            ok_('a2' in h5.keys())
            assert_allclose(h5['a2'][...],
                            correct_result)

        path = os.path.join(folder, 'projectionizer_test_write_synapses_efferent.h5')
        write_nrn.write_synapses(path, df.groupby('sgid'),
                                 SYNAPSE_PARAMS, efferent=True)

        correct_result = [[1.00000000e+00, 1.00000000e+00, 1.03300000e+03, 1.03300000e+03,
                           3.75173779e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                           5.80177543e+00, 5.08346474e+01, 4.77665138e+01, 1.46902748e+02,
                           7.25006104e+01, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ]]

        with h5py.File(path) as h5:
            ok_('a2' in h5.keys())
            assert_allclose(h5['a2'][...],
                            correct_result)


def _write_synapses(synapse_data={}, synapse_params={}, efferent=False):
    """Utility function to create synapses data and call write_nrn.write_synapses"""

    np.random.seed(1337)
    data = {'tgid': [1],
            'sgid': [2],
            'y': [0.33],
            'section_id': [1033],
            'location': [3],
            'syn_ids': [12],
            'sgid_path_distance': [300.],
            'segment_id': [1033]}
    data.update(synapse_data)

    df = pd.DataFrame(data)

    _synapse_params = SYNAPSE_PARAMS.copy()
    _synapse_params.update(synapse_params)

    with setup_tempdir('test_utils') as folder:
        path = os.path.join(folder, 'projectionizer_test_write_synapses.h5')
        write_nrn.write_synapses(path, df.groupby('sgid'),
                                 _synapse_params, efferent=efferent)


def test_write_conform_synapses():
    _write_synapses({})


@raises(AssertionError)
def test_write_non_conform_location():
    _write_synapses({'location': [-3]})


@raises(AssertionError)
def test_write_non_conform_sgid():
    _write_synapses({'sgid': [-3]})


@raises(AssertionError)
def test_write_non_conform_tgid():
    _write_synapses({'tgid': [-3]}, efferent=True)


@raises(AssertionError)
def test_write_non_conform_syn_type():
    _write_synapses(synapse_params={'id': [-3]})


@raises(ValueError)
def test_write_non_conform_D_synapse_param():
    _write_synapses(synapse_params={'D': (-7, 8)})


@raises(ValueError)
def test_write_non_conform_F_synapse_param():
    _write_synapses(synapse_params={'F': (-7, -8)})


@raises(ValueError)
def test_write_non_conform_DTC_synapse_param():
    _write_synapses(synapse_params={'DTC': (-7, -8)})
