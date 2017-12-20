import h5py
import luigi
import numpy as np
import pandas as pd
from luigi.contrib.simulate import RunAnywayTarget
from mock import patch
from nose.tools import ok_, raises
from numpy.testing import assert_allclose, assert_equal

from projectionizer.step_3_write import *
from utils import setup_tempdir

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
                       'segment_id': [1033]})

    assert_allclose(create_synapse_data(df, SYNAPSE_PARAMS, efferent=True),
                    [[1.00000000e+00,   1.10000000e-03,   1.03300000e+03,   1.03300000e+03,
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
                       'segment_id': [1033]})

    with setup_tempdir('test_utils') as folder:
        path = os.path.join(folder, 'projectionizer_test_write_synapses.h5')
        write_synapses(path, df.groupby('sgid'),
                       SYNAPSE_PARAMS, efferent=False)

        f = h5py.File(path)
        ok_('a2' in f.keys())
        assert_allclose(f['a2'][...],
                        [[2.00000000e+00,   1.10000000e-03,   1.03300000e+03,   1.03300000e+03,
                          -7.86074025e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                          7.75316318e+00,   1.92565485e+01,   2.51811971e+01,   9.06247588e+01,
                          1.17854003e+02,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                          0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ]])

        path = os.path.join(folder, 'projectionizer_test_write_synapses_efferent.h5')
        write_synapses(path, df.groupby('sgid'),
                       SYNAPSE_PARAMS, efferent=True)

        f = h5py.File(path)
        ok_('a2' in f.keys())
        assert_allclose(f['a2'][...],
                        [[1.00000000e+00,   1.10000000e-03,   1.03300000e+03,   1.03300000e+03,
                          -3.75173779e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                          5.80177543e+00,   5.08346474e+01,   4.77665138e+01,   1.46902748e+02,
                          7.25006104e+01,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                          0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])


def test_write_summary():
    with setup_tempdir('test_utils') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        pd.DataFrame({'tgid': [1],
                      'sgid': [2],
                      'y': [0.33],
                      'section_id': [1033],
                      'location': [-3],
                      'syn_ids': [12],
                      'segment_id': [1033]}).to_feather(mock_path)

        class TestWriteSummary(WriteSummary):
            # Dummy values
            folder = tmp_folder
            circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = voxel_path = prefix = extension = None

            def input(self):
                class Return:
                    path = mock_path
                return Return

        test = TestWriteSummary()
        test.run()


def _raise_exception(path, itr):
    raise OSError()


@raises(OSError)
@patch('projectionizer.step_3_write.write_synapses_summary', _raise_exception)
def test_write_summary_raises():
    with setup_tempdir('test_utils') as folder:
        mock_path = os.path.join(folder, 'mock_synapses.feather')
        pd.DataFrame({'tgid': [1],
                      'sgid': [2],
                      'y': [0.33],
                      'section_id': [1033],
                      'location': [-3],
                      'syn_ids': [12],
                      'segment_id': [1033]}).to_feather(mock_path)

        class TestWriteSummary(WriteSummary):
            # Dummy values
            folder = circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = voxel_path = prefix = extension = None

            def input(self):
                class Return:
                    path = mock_path
                return Return

        test = TestWriteSummary()
        test.run()


def test_get_synapse_parameters():
    class TestWriteNrnH5(WriteNrnH5):
        folder = circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = voxel_path = prefix = extension = None
        synapse_type = gsyn_mean = gsyn_sigma = use_mean = use_sigma = D_mean = 1
        D_sigma = F_mean = F_sigma = DTC_mean = DTC_sigma = ASE_mean = ASE_sigma = 1

    assert_equal(TestWriteNrnH5().get_synapse_parameters(),
                 {'Use': (1, 1), 'D': (1, 1), 'F': (1, 1), 'DTC': (1, 1), 'gsyn': (1, 1), 'Ase': (1, 1), 'id': 1})
