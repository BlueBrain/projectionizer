import os
import shutil

from luigi import Task
from mock import Mock, patch
from nose.tools import eq_, ok_, assert_raises
from numpy.testing import assert_equal
import pandas as pd

from projectionizer import step_3_write

from utils import setup_tempdir, TEST_DATA_DIR


def test_write_summary():
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        pd.DataFrame({'tgid': [1],
                      'sgid': [2],
                      'y': [0.33],
                      'section_id': [1033],
                      'location': [-3],
                      'syn_ids': [12],
                      'segment_id': [1033]}).to_feather(mock_path)
        mock = Mock(path=mock_path)

        class TestWriteSummary(step_3_write.WriteSummary):
            circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = None
            voxel_path = prefix = extension = None
            folder = tmp_folder
            layers = ''

            def input(self):
                return mock

        test = TestWriteSummary()
        ok_(isinstance(test.requires(), Task))

        test.run()
        output_path = os.path.join(tmp_folder, 'proj_nrn_summary.h5')
        ok_(os.path.exists(output_path))

        with patch('projectionizer.step_3_write.write_synapses_summary', Mock(side_effect=OSError)):
            test = TestWriteSummary()
            assert_raises(OSError, test.run)


def test_WriteNrnH5():
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        pd.DataFrame({'tgid': [1],
                      'sgid': [2],
                      'section_id': [1033],
                      'segment_id': [1033],
                      'location': [0.5],
                      'sgid_path_distance': [0.5],
                      }).to_feather(mock_path)
        mock = Mock(path=mock_path)

        class TestWriteNrnH5(step_3_write.WriteNrnH5):
            efferent = False
            folder = tmp_folder
            circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = None
            voxel_path = prefix = extension = None
            synapse_type = gsyn_mean = gsyn_sigma = use_mean = use_sigma = D_mean = 1
            D_sigma = F_mean = F_sigma = DTC_mean = DTC_sigma = ASE_mean = ASE_sigma = 1
            layers = ''

            def input(self):
                return mock

        assert_equal(TestWriteNrnH5().get_synapse_parameters(),
                     {'Use': (1, 1),
                      'D': (1, 1),
                      'F': (1, 1),
                      'DTC': (1, 1),
                      'gsyn': (1, 1),
                      'Ase': (1, 1),
                      'id': 1,
                      })

        test = TestWriteNrnH5()
        ok_(isinstance(test.requires(), Task))

        test.run()
        output_path = os.path.join(tmp_folder, 'proj_nrn.h5')
        ok_(os.path.exists(output_path))

        with patch('projectionizer.step_3_write.write_synapses', Mock(side_effect=OSError)):
            test = TestWriteNrnH5()
            assert_raises(OSError, test.run)


def test_WriteUserTargetTxt():
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        pd.DataFrame({'tgid': [1],
                      'sgid': [2],
                      'section_id': [1033],
                      'segment_id': [1033],
                      'location': [0.5],
                      'sgid_path_distance': [0.5],
                      }).to_feather(mock_path)
        mock = Mock(path=mock_path)

        class TestWriteUserTargetTxt(step_3_write.WriteUserTargetTxt):
            efferent = False
            folder = tmp_folder
            circuit_config = geometry = n_total_chunks = sgid_offset = oversampling = None
            voxel_path = prefix = extension = None
            layers = ''

            def input(self):
                return mock

        test = TestWriteUserTargetTxt()
        ok_(isinstance(test.requires(), Task))

        test.run()
        output_path = os.path.join(tmp_folder, 'user.target')
        ok_(os.path.exists(output_path))


def test_VirtualFibers():
    data = os.path.join(TEST_DATA_DIR, 'virtual-fibers-no-offset.csv')
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'virtual-fibers-no-offset.csv')
        shutil.copyfile(data, mock_path)
        mock = Mock(path=mock_path)

        class TestVirtualFibers(step_3_write.VirtualFibers):
            folder = tmp_folder
            sgid_offset = 10
            circuit_config = geometry = n_total_chunks = oversampling = None
            voxel_path = prefix = None
            layers = ''

            def input(self):
                return mock

        test = TestVirtualFibers()
        ok_(isinstance(test.requires(), Task))

        test.run()
        output_path = os.path.join(tmp_folder, 'test-virtual-fibers.csv')
        ok_(os.path.exists(output_path))
        df = pd.read_csv(output_path)
        eq_(TestVirtualFibers.sgid_offset, df.sgid.min())
