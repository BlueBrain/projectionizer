import os
import numpy as np
import numpy.testing as npt
import pandas as pd
from mock import Mock

from nose.tools import ok_, eq_

from projectionizer import step_0_sample
from projectionizer.luigi_utils import CommonParams
from projectionizer.utils import load, write_feather

from utils import setup_tempdir

def test_SampleChunk():
    with setup_tempdir('test_step_0') as tmp_folder:
        mock_path = os.path.join(tmp_folder, 'full-sample.feather')
        write_feather(mock_path, pd.DataFrame(np.arange(100), columns=['foo', ]))
        params = {'circuit_config': 'circuit',
                  'folder': tmp_folder,
                  'sgid_offset': 0,
                  'oversampling': 0,
                  'voxel_path': '',
                  'layers': '',
                  }
        mock = Mock(path=mock_path)
        class TestSampleChunk(step_0_sample.SampleChunk):
            n_slices = 1
            def input(self):
                return mock

        task = TestSampleChunk(n_total_chunks=5, chunk_num=0, **params)
        task.run()
        #load first one
        output = os.path.join(tmp_folder, 'test-sample-chunk-0.feather')
        chunked = load(output)
        eq_(len(chunked), 21)
        eq_(chunked.foo.iloc[0], 0)
        eq_(chunked.foo.iloc[-1], 20)

        #load last one
        task = TestSampleChunk(n_total_chunks=5, chunk_num=4, **params)
        task.run()
        output = os.path.join(tmp_folder, 'test-sample-chunk-4.feather')
        chunked = load(output)
        eq_(len(chunked), 16)
        eq_(chunked.foo.iloc[0], 84)
        eq_(chunked.foo.iloc[-1], 99)

        ok_(isinstance(task.requires(), CommonParams))
