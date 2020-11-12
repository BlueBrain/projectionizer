import os
import shutil
import subprocess

import h5py
from luigi import Task
from mock import Mock, patch
from nose.tools import eq_, ok_, assert_raises
import numpy as np
from numpy.testing import assert_equal
import pandas as pd

from projectionizer import step_3_write
from projectionizer.utils import write_feather

from utils import (setup_tempdir,
                   EDGE_POPULATION,
                   NODE_POPULATION,
                   TEST_DATA_DIR)
import logging
logging.basicConfig()
L = logging.getLogger(__name__)

def test_write_summary():
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        data = {'tgid': [1],
                'sgid': [2],
                'y': [0.33],
                'section_id': [1033],
                'synapse_offset': [128.],
                'afferent_indices': [12],
                'segment_id': [1033],
                }
        write_feather(mock_path, pd.DataFrame(data))
        mock = Mock(path=mock_path)

        class TestWriteSummary(step_3_write.WriteSummary):
            n_total_chunks = sgid_offset = oversampling = None
            circuit_config = extension = 'fake_string'
            folder = tmp_folder
            layers = ''

            def input(self):
                return mock

        test = TestWriteSummary()
        ok_(isinstance(test.requires(), Task))

        test.run()
        output_path = os.path.join(tmp_folder, 'proj_nrn_summary.h5')
        ok_(os.path.exists(output_path))

        with patch('projectionizer.step_3_write.write_nrn') as write_nrn:
            write_nrn.write_synapses_summary = Mock(side_effect=OSError)
            test = TestWriteSummary()
            assert_raises(OSError, test.run)


def test_WriteNrnH5():
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        data = {'tgid': [1],
                'sgid': [2],
                'section_id': [1033],
                'segment_id': [1033],
                'synapse_offset': [128.],
                'sgid_path_distance': [0.5],
                }
        write_feather(mock_path, pd.DataFrame(data))
        mock = Mock(path=mock_path)

        class TestWriteNrnH5(step_3_write.WriteNrnH5):
            efferent = False
            folder = tmp_folder
            circuit_config = 'fake_string'
            geometry = n_total_chunks = sgid_offset = oversampling = None
            extension = None
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
                      'ASE': (1, 1),
                      'id': 1,
                      })

        test = TestWriteNrnH5()
        ok_(isinstance(test.requires(), Task))

        test.run()
        output_path = os.path.join(tmp_folder, 'proj_nrn.h5')
        ok_(os.path.exists(output_path))

        with patch('projectionizer.step_3_write.write_nrn') as write_nrn:
            write_nrn.write_synapses = Mock(side_effect=OSError)
            test = TestWriteNrnH5()
            assert_raises(OSError, test.run)


def test_WriteUserTargetTxt():
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        data = {'tgid': [1],
                'sgid': [2],
                'section_id': [1033],
                'segment_id': [1033],
                'synapse_offset': [128.],
                'sgid_path_distance': [0.5],
                }
        write_feather(mock_path, pd.DataFrame(data))
        mock = Mock(path=mock_path)

        class TestWriteUserTargetTxt(step_3_write.WriteUserTargetTxt):
            efferent = False
            folder = tmp_folder
            circuit_config = 'fake_string'
            geometry = n_total_chunks = sgid_offset = oversampling = None
            extension = None
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
            circuit_config = 'fake_string'
            geometry = n_total_chunks = oversampling = None
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


def test_WriteSonata():
    with setup_tempdir('test_step3') as tmp_folder:
        syns_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        df = pd.DataFrame({'tgid': [10],
                           'sgid': [20],
                           'section_id': [1033],
                           'segment_id': [1033],
                           'synapse_offset': [128.],
                           'x': [101],
                           'y': [102],
                           'z': [103],
                           'sgid_path_distance': [0.5],
                           })
        write_feather(syns_path, df)
        mock_feather = Mock(path=syns_path)

        def create_h5_files(sonata, node, edge):
            with h5py.File(sonata, 'w') as h5:
                group = h5.create_group(f'edges/{EDGE_POPULATION}')
                group['source_node_id'] = [0] * len(df.sgid)
            with h5py.File(node, 'w') as h5:
                group = h5.create_group(f'nodes/{NODE_POPULATION}')
                group['node_type_id'] = np.full(df.sgid.max(), -1)
            with h5py.File(edge, 'w') as h5:
                group = h5.create_group(f'edges/{EDGE_POPULATION}')
                group['source_node_id'] = df.sgid.to_numpy() - 1
                group['target_node_id'] = df.tgid.to_numpy() - 1


        class TestWriteSonata(step_3_write.WriteSonata):
            folder = tmp_folder
            recipe_path = morphology_path = circuit_config = 'fake_string'
            sgid_offset = geometry = n_total_chunks = oversampling = None
            node_population = NODE_POPULATION
            edge_population = EDGE_POPULATION
            node_file_name = 'nodes.h5'
            edge_file_name = 'edges.h5'
            layers = ''

            def input(self):
                sonata_path = self._get_full_path_output(self.edge_file_name)
                node_path = self._get_full_path_output(self.node_file_name)
                edge_path = self.clone(step_3_write.WriteSonataEdges).output().path
                return (Mock(path=sonata_path),
                        mock_feather,
                        Mock(path=node_path),
                        Mock(path=edge_path),
                        None)

        test = TestWriteSonata()
        assert len(test.requires()) == 5
        assert all(isinstance(t, Task) for t in test.requires())
        assert test.requires()[4].target_name == test.mtype
        assert_equal(test.requires()[0].output().path, test.output().path)

        sonata_path = test.input()[0].path
        node_path = test.input()[2].path
        edge_path = test.input()[3].path

        # Clean run
        create_h5_files(sonata_path, node_path, edge_path)
        test.run()

        # Wrong edge count
        with h5py.File(sonata_path, 'r+') as h5:
            del h5[f'edges/{EDGE_POPULATION}/source_node_id']
            h5[f'edges/{EDGE_POPULATION}/source_node_id'] = [0] * (len(df.sgid) + 1)
        assert_raises(AssertionError, test.run)

        # Wrong node count
        create_h5_files(sonata_path, node_path, edge_path)
        with h5py.File(node_path, 'r+') as h5:
            del h5[f'nodes/{NODE_POPULATION}/node_type_id']
            h5[f'nodes/{NODE_POPULATION}/node_type_id'] = np.full(df.sgid.max() + 1, -1)
        assert_raises(AssertionError, test.run)

        # SGIDs are off
        create_h5_files(sonata_path, node_path, edge_path)
        with h5py.File(edge_path, 'r+') as h5:
            del h5[f'edges/{EDGE_POPULATION}/source_node_id']
            h5[f'edges/{EDGE_POPULATION}/source_node_id'] = df.sgid.to_numpy()
        assert_raises(AssertionError, test.run)

        # TGIDs are off
        create_h5_files(sonata_path, node_path, edge_path)
        with h5py.File(edge_path, 'r+') as h5:
            del h5[f'edges/{EDGE_POPULATION}/target_node_id']
            h5[f'edges/{EDGE_POPULATION}/target_node_id'] = df.tgid.to_numpy()
        assert_raises(AssertionError, test.run)


def test_WriteSonataNodes():
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        data = {'tgid': [10],
                'sgid': [20],
                'section_id': [1033],
                'segment_id': [1033],
                'synapse_offset': [128.],
                'x': [101],
                'y': [102],
                'z': [103],
                'sgid_path_distance': [0.5],
                }
        write_feather(mock_path, pd.DataFrame(data))
        mock = Mock(path=mock_path)

        class TestWriteSonataNodes(step_3_write.WriteSonataNodes):
            folder = tmp_folder
            recipe_path = morphology_path = circuit_config = 'fake_string'
            sgid_offset = geometry = n_total_chunks = oversampling = None
            node_population = NODE_POPULATION
            edge_population = EDGE_POPULATION
            node_file_name = 'nodes.h5'
            layers = ''

            def input(self):
                return mock

        test = TestWriteSonataNodes()
        assert isinstance(test.requires(), Task)

        test.run()
        assert os.path.isfile(test.output().path)

        with h5py.File(test.output().path, 'r') as h5:
            assert_equal(len(h5[f'nodes/{NODE_POPULATION}/node_type_id']), data['sgid'][0])


def test_WriteSonataEdges():
    with setup_tempdir('test_step3') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'mock_synapses.feather')
        data = {'tgid': [10],
                'sgid': [20],
                'section_id': [1033],
                'segment_id': [1033],
                'synapse_offset': [128.],
                'x': [101],
                'y': [102],
                'z': [103],
                'sgid_path_distance': [0.5],
                }
        write_feather(mock_path, pd.DataFrame(data))
        mock = Mock(path=mock_path)

        class TestWriteSonataEdges(step_3_write.WriteSonataEdges):
            folder = tmp_folder
            recipe_path = morphology_path = circuit_config = 'fake_string'
            sgid_offset = geometry = n_total_chunks = oversampling = None
            node_population = NODE_POPULATION
            edge_population = EDGE_POPULATION
            layers = ''
            edge_file_name = 'edges.h5'

            def input(self):
                return mock

        test = TestWriteSonataEdges()
        assert isinstance(test.requires(), Task)

        test.run()
        assert os.path.isfile(test.output().path)

        with h5py.File(test.output().path, 'r') as h5:
            assert_equal(h5[f'edges/{EDGE_POPULATION}/source_node_id'][0], data['sgid'][0] - 1)
            assert_equal(h5[f'edges/{EDGE_POPULATION}/target_node_id'][0], data['tgid'][0] - 1)


def test_check_if_old_syntax():
    archive = 'fake_archive'
    res = step_3_write._check_if_old_syntax(archive)
    assert_equal(res, False)

    archive = 'archive/2021-07'
    res = step_3_write._check_if_old_syntax(archive)
    assert_equal(res, False)

    archive = 'archive/2021-06'
    res = step_3_write._check_if_old_syntax(archive)
    assert_equal(res, True)

    archive = 'archive/2020-12'
    res = step_3_write._check_if_old_syntax(archive)
    assert_equal(res, True)


def test_RunSpykfunc():
    with setup_tempdir('test_step3') as tmp_folder:
        class TestRunSpykfunc(step_3_write.RunSpykfunc):
            folder = tmp_folder
            recipe_path = morphology_path = circuit_config = 'fake_string'
            sgid_offset = geometry = n_total_chunks = oversampling = None
            layers = ''

            def run(self):
                os.makedirs(self.output().path, exist_ok=True)
                super().run()

        with patch('projectionizer.step_3_write.subprocess.run') as patched:

            test = TestRunSpykfunc()
            assert len(test.requires()) == 2
            assert all(isinstance(t, Task) for t in test.requires())

            test.run()
            assert os.path.isdir(os.path.join(tmp_folder, 'spykfunc'))

            # Test that the spykfunc dir is removed on error
            patched.side_effect = subprocess.CalledProcessError(1, 'fake')
            assert_raises(subprocess.CalledProcessError, test.run)
            assert not os.path.isdir(os.path.join(tmp_folder, 'spykfunc'))

        with patch('projectionizer.step_3_write.subprocess.run'):
            test = TestRunSpykfunc()
            test.module_archive = 'unstable'
            command = test._parse_command()
            assert '--touches' not in command

            test.module_archive = 'archive/2020-12'
            command = test._parse_command()
            assert '--touches' in command


def test_RunParquetConverter():
    with setup_tempdir('test_step3') as tmp_folder:
        class TestRunParquetConverter(step_3_write.RunParquetConverter):
            folder = tmp_folder
            recipe_path = morphology_path = circuit_config = 'fake_string'
            sgid_offset = geometry = n_total_chunks = oversampling = None
            layers = ''

            def run(self):
                with open(self.output().path, 'w') as fd:
                    fd.write('')
                super().run()

        with patch('projectionizer.step_3_write.subprocess.run'):
            test = TestRunParquetConverter()
            assert len(test.requires()) == 2
            assert all(isinstance(t, Task) for t in test.requires())

            test.run()
            assert os.path.isfile(os.path.join(tmp_folder, test.edge_file_name))

            test_file_name = 'fake.sonata'
            test.edge_file_name = test_file_name
            test.run()
            assert os.path.isfile(os.path.join(tmp_folder, test_file_name))

        with patch('projectionizer.step_3_write.subprocess.run'):
            test = TestRunParquetConverter()
            test.module_archive = 'unstable'
            command = test._parse_command()
            assert '--format' not in command

            test.module_archive = 'archive/2020-12'
            command = test._parse_command()
            assert '--format' in command
