import numpy as np
import pandas as pd

from projectionizer import step_2_prune


def test_find_cutoff_mean_per_mtype():
    value_count = pd.Series(np.arange(10))
    assert step_2_prune.find_cutoff_mean_per_mtype(value_count, 0.4) == 6.469387755102041
    assert step_2_prune.find_cutoff_mean_per_mtype(value_count, 0.5) == 7.0390625
    assert step_2_prune.find_cutoff_mean_per_mtype(value_count, 0.6) == 7.484375


def test_calculate_cutoff_means():
    columns = ["mtype", "sgid", "tgid", "connection_size"]
    mtype_sgid_tgid = pd.DataFrame(
        [
            ["mtype0", 0, 1, 10],
            ["mtype0", 1, 2, 5],
            ["mtype0", 2, 3, 1],
            ["mtype1", 5, 6, 10],
        ],
        columns=columns,
    )
    ret = step_2_prune.calculate_cutoff_means(mtype_sgid_tgid, oversampling=2.0)
    expected = pd.DataFrame(
        {
            "mtype": pd.Categorical(["mtype0", "mtype1"]),
            "cutoff": [6.0, 10.0],
        }
    )
    pd.testing.assert_frame_equal(expected, ret)
