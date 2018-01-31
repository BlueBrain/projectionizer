import numpy as np
import pandas as pd
from numpy.testing import assert_equal

from projectionizer.step_2_prune import *


def test_find_cutoff_mean_per_mtype():
    value_count = pd.Series(np.arange(10))
    assert_equal(find_cutoff_mean_per_mtype(value_count, 0.4), 6.469387755102041)
    assert_equal(find_cutoff_mean_per_mtype(value_count, 0.5), 7.0390625)
    assert_equal(find_cutoff_mean_per_mtype(value_count, 0.6), 7.484375)
