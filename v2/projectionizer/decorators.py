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


def _hash(obj):
    # try:
    import bluepy.v2.circuit
    if isinstance(obj, bluepy.v2.circuit.Circuit):
        ret = hash(tuple(sorted(obj._config)))
        return ret
    if isinstance(obj, pd.DataFrame):
        ret = hashlib.md5(obj.values.tobytes()).hexdigest()
        return ret
    if isinstance(obj, (list, tuple, numpy.ndarray)):
        ret = hash(tuple(_hash(o) for o in obj))
        return ret
    ret = hash(obj)
    return hash(obj)


# -3931883067604897208
# -963337948087315334
# -3931883067604897208
# -963337948087315334
# DEBUG:root:Hash is : 4614553062794121523

# (-3931883067604897208, 8728498274001, 10, -963337948087315334, 363)


def simple_cache(f):
    """A simple cache that just caches based on the function name"""
    @wraps(f)
    def wrapper(*args, **kwgs):
        CACHE_DIR = os.path.join(os.getenv('HOME'), '.data_cache')
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)
        cached_filename = os.path.join(CACHE_DIR, '{}'.format(f.__name__))
        if os.path.exists(cached_filename):
            L.debug('Reading from cache')
            return pd.read_feather(cached_filename)
        else:
            L.debug('Cache miss')
            res = f(*args, **kwgs)
            try:
                _write_feather(cached_filename, res)
            except Exception as e:
                L.warning('Failed to cache file. Reason: {}'.format(e))
                try:
                    os.remove(cached_filename)
                except:
                    pass
            return res
    return wrapper


def pandas_cache(f):
    @wraps(f)
    def wrapper(*args, **kwgs):

        CACHE_DIR = os.path.join(os.getenv('HOME'), '.data_cache')
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)

        try:
            base = tuple(map(_hash, chain(args, sorted(kwgs.values()), inspect.getsourcelines(f))))
            print(base)
            h = hash(base)

        except TypeError as e:
            error_msg = "{}() arguments must be hashable to use the 'cache' decorator"\
                .format(f.__name__)
            error_msg += "\n{}".format(str(e))
            raise Exception(
                error_msg)

        L.debug('Hash is : {}'.format(h))
        cached_filename = os.path.join(CACHE_DIR, '{}-{}'.format(f.__name__, h))
        if os.path.exists(cached_filename):
            L.debug('Reading from cache')
            return pd.read_feather(cached_filename)
        else:
            L.debug('Cache miss')
            res = f(*args, **kwgs)
            try:
                _write_feather(cached_filename, res)
            except:
                try:
                    os.remove(cached_filename)
                except:
                    pass
            return res

    return wrapper


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    @timeit('blah')
    def hello(*args, **kwargs):
        return pd.DataFrame([1, 2, 3])

    hello(1, [1, 2, 3])


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
