import os

import pandas as pd
from mock import patch
from nose.tools import ok_, raises
from numpy.testing import assert_allclose, assert_equal

from projectionizer import step_3_write
from projectionizer.tests.utils import setup_tempdir


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

        class TestWriteSummary(step_3_write.WriteSummary):
            # Dummy values
            folder = circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = None
            voxel_path = prefix = extension = None

            def input(self):
                class Return:
                    path = mock_path
                return Return

        test = TestWriteSummary()
        test.run()


def test_get_synapse_parameters():
    class TestWriteNrnH5(step_3_write.WriteNrnH5):
        folder = circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = None
        voxel_path = prefix = extension = None
        synapse_type = gsyn_mean = gsyn_sigma = use_mean = use_sigma = D_mean = 1
        D_sigma = F_mean = F_sigma = DTC_mean = DTC_sigma = ASE_mean = ASE_sigma = 1

    assert_equal(TestWriteNrnH5().get_synapse_parameters(),
                 {'Use': (1, 1),
                  'D': (1, 1),
                  'F': (1, 1),
                  'DTC': (1, 1),
                  'gsyn': (1, 1),
                  'Ase': (1, 1),
                  'id': 1,
                  })


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

        class TestWriteSummary(step_3_write.WriteSummary):
            # Dummy values
            folder = tmp_folder
            circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = None
            voxel_path = prefix = extension = None

            def input(self):
                class Return:
                    path = mock_path
                return Return

        test = TestWriteSummary()
        test.run()
