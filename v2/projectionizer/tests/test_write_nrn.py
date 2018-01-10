import os
from utils import setup_tempdir

import h5py
import numpy as np
import pandas as pd

from projectionizer import write_nrn
from numpy.testing import assert_allclose, assert_equal
from nose.tools import ok_, raises


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
                       'sgid_distance': [300.]
                       })

    assert_allclose(write_nrn.create_synapse_data(df, SYNAPSE_PARAMS, efferent=True),
                    [[1.00000000e+00,   300./300.,   1.03300000e+03,   1.03300000e+03,
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
                       'location': [-3],
                       'syn_ids': [12],
                       'segment_id': [1033],
                       'sgid_distance': [300.]
                       })

    with setup_tempdir('test_utils') as folder:
        path = os.path.join(folder, 'projectionizer_test_write_synapses.h5')
        write_nrn.write_synapses(path, df.groupby('sgid'),
                                 SYNAPSE_PARAMS, efferent=False)

        with h5py.File(path) as h5:
            ok_('a2' in h5.keys())
            assert_allclose(h5['a2'][...],
                            [[2.00000000e+00,   300./300.,   1.03300000e+03,   1.03300000e+03,
                              -7.86074025e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                              7.75316318e+00,   1.92565485e+01,   2.51811971e+01,   9.06247588e+01,
                              1.17854003e+02,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                              0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ]])

        path = os.path.join(folder, 'projectionizer_test_write_synapses_efferent.h5')
        write_nrn.write_synapses(path, df.groupby('sgid'),
                                 SYNAPSE_PARAMS, efferent=True)

        with h5py.File(path) as h5:
            ok_('a2' in h5.keys())
            assert_allclose(h5['a2'][...],
                            [[1.00000000e+00,   300./300.,   1.03300000e+03,   1.03300000e+03,
                              -3.75173779e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                              5.80177543e+00,   5.08346474e+01,   4.77665138e+01,   1.46902748e+02,
                              7.25006104e+01,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                              0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])
