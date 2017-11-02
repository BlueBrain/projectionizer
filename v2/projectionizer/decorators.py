import hashlib
import inspect
import logging
import os
from datetime import datetime
from functools import wraps
from itertools import chain

import numpy
import pandas as pd

from projectionizer import utils

L = logging.getLogger()


def timeit(name):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            start = datetime.now()
            L.debug('Starting Name: %s time: %s', name, start)
            res = f(*args, **kwds)
            L.debug('Name: %s time: %s', name, datetime.now() - start)
            return res
        return wrapper
    return decorator


def _write_feather(name, df):
    df = df.copy()
    df.columns = map(str, df.columns)
    df = df.reset_index(drop=True)
    df.to_feather(name)


def load_feather(name):
    synapses = pd.read_feather(name)
    # START_COLS = map(str, utils.SEGMENT_START_COLS)
    # END_COLS = map(str, utils.SEGMENT_END_COLS)
    # for k, st, en in zip('xyz', START_COLS, END_COLS):
    #     synapses[k] = (synapses[st] + synapses[en]) / 2.
    # synapses.index = synapses['index']
    # synapses.drop(['index'] + START_COLS + END_COLS,
    #               axis=1, inplace=True)
    return synapses
