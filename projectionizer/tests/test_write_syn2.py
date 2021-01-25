import functools
import os
import shutil

import h5py
import numpy as np
import pandas as pd

from nose.tools import eq_, ok_
from numpy.testing import assert_allclose, assert_raises

from projectionizer import write_syn2
from utils import setup_tempdir


assert_allclose = functools.partial(assert_allclose, rtol=0.0001, atol=0.0001)


def test_write_synapses():
    count = 5
    syns = pd.DataFrame({
        'tgid': np.random.randint(10, size=(count, )),
        'sgid': np.random.randint(10, size=(count, )),
        'delay': np.random.random(size=(count, )),
        })


    datasets_types = {'delay': np.float32,
                      'connected_neurons_pre': np.int,
                      'connected_neurons_post': np.int,
                      }

    def _fake_create_synapse_data(name, dtype, syns):
        name, dtype = name, dtype
        return [1.] * len(syns)

    with setup_tempdir('test_write_syn2') as tmp:
        output_path = os.path.join(tmp, 'fake.syn2')

        write_syn2.write_synapses(syns,
                                  output_path,
                                  _fake_create_synapse_data,
                                  datasets_types)

        ok_(os.path.exists(output_path))

        with h5py.File(output_path, 'r') as h5:
            props = h5[write_syn2.DEFAULT_GROUP]
            ok_('delay' in props)
            eq_(list(props['delay']), [1.] * 5)



def test__distribute_param():
    np.random.seed(0)
    syns = pd.DataFrame({
        'synapse_type_name': [0, 0, 0, 1, 1],
        })

    synapse_data = {
        'type_0':
        {'physiology':
         {'gsyn': {'distribution': {'name': 'truncated_gaussian',
                                    'params': {'mean': 1., 'std': 1.}}},
          'nrrp': {'distribution': {'name': 'uniform_int',
                                    'params': {'min': 0, 'max': 10}}},
          'u_hill_coefficient': {'distribution': {'name': 'fixed_value',
                                                  'params': {'value': 2.79}}}
          },
         },
        'type_1':
        {'physiology':
         {'gsyn': {'distribution': {'name': 'truncated_gaussian',
                                    'params': {'mean': 0.1, 'std': 0.1}}},
          'nrrp': {'distribution': {'name': 'uniform_int',
                                    'params': {'min': 10, 'max': 20}}},
          'u_hill_coefficient': {'distribution': {'name': 'fixed_value',
                                                  'params': {'value': 2.79}}}
          },
         },
        }

    ret = write_syn2._distribute_param(syns, synapse_data, prop='conductance', dtype=np.float)
    assert_allclose(ret, [0.022722, 1.400157, 1.978738, 0.195009, 0.084864])

    ret = write_syn2._distribute_param(syns, synapse_data, prop='n_rrp_vesicles', dtype=np.int)
    assert_allclose(ret, [8, 10, 1, 16, 17])

    with assert_raises(write_syn2.OptionalParameterException):
        ret = write_syn2._distribute_param(syns, synapse_data, prop='conductance_scale_factor', dtype=np.float)

    ret = write_syn2._distribute_param(syns, synapse_data, prop='u_hill_coefficient', dtype=np.float)
    assert_allclose(ret, np.ones(len(syns)) * 2.79)


def test_create_synapse_data():
    np.random.seed(0)

    count = 5
    syns = pd.DataFrame({
        'tgid': np.random.randint(10, size=(count, )),
        'sgid': np.random.randint(10, size=(count, )),
        'delay': np.random.random(size=(count, )),
        'x': np.random.random(size=(count, )),
        'gsyn': np.random.random(size=(count, )),
        })

    ret = write_syn2.create_synapse_data('connected_neurons_pre', np.int, syns, {})
    assert_allclose(ret, [8, 2, 4, 1, 3])

    ret = write_syn2.create_synapse_data('delay', np.float, syns, {})
    assert_allclose(ret, [0.297535, 0.056713, 0.272656, 0.477665, 0.812169], )

    ret = write_syn2.create_synapse_data('syn_type_id', np.int, syns, {})
    assert_allclose(ret, [120.] * count)

    #something from DATASETS_DIRECT_MAP, maps to 'x'
    ret = write_syn2.create_synapse_data('afferent_center_x', np.float, syns, {})
    assert_allclose(ret, [0.479977, 0.392785, 0.836079, 0.337396, 0.648172])

def test__truncated_gaussian():
    np.random.seed(0)
    mean, std, count = 1., 5., 1000

    ret = write_syn2._truncated_gaussian(mean, std, count, truncated_max_stddev=1.)
    eq_(len(ret), count)
    ok_(np.all(ret > 0.))
    ok_(np.all(ret < mean + 1. * std))

    ret = write_syn2._truncated_gaussian(mean, std, count, truncated_max_stddev=10.)
    eq_(len(ret), count)
    ok_(np.all(ret > 0.))

    ok_(np.all(ret < mean + 10. * std))

