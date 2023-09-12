from unittest.mock import Mock

import luigi
import numpy as np
import pandas as pd
import pytest

import projectionizer
import projectionizer.step_0_sample as test_module


@pytest.mark.MockTask(cls=test_module.SampleChunk)
def test_SampleChunk(MockTask):
    mock_path = MockTask.folder / "full-sample.feather"
    projectionizer.utils.write_feather(mock_path, pd.DataFrame(np.arange(100), columns=["foo"]))

    class TestSampleChunk(MockTask):
        n_total_chunks = 5
        chunk_num = luigi.IntParameter()

        def input(self):
            return Mock(path=mock_path)

    task = TestSampleChunk(chunk_num=0)
    task.run()
    # load first one
    output = task.folder / "test-sample-chunk-0.feather"
    chunked = projectionizer.utils.load(output)
    assert len(chunked) == 21
    assert chunked.foo.iloc[0] == 0
    assert chunked.foo.iloc[-1] == 20

    # load last one
    task = TestSampleChunk(chunk_num=4)
    task.run()
    output = task.folder / "test-sample-chunk-4.feather"
    chunked = projectionizer.utils.load(output)
    assert len(chunked) == 16
    assert chunked.foo.iloc[0] == 84
    assert chunked.foo.iloc[-1] == 99

    assert isinstance(task.requires(), projectionizer.luigi_utils.CommonParams)
