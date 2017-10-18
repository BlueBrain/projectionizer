import logging
from itertools import chain, repeat

import nose.tools as nt
import numpy.testing as npt
from mock import MagicMock
from nose.tools import eq_, ok_, raises

from examples.csThalamocortical_VPM_tcS2F_2p6_ps import *
from examples.mini_col_locations import tiled_locations

L = logging.getLogger(__name__)


def test_grid_locations():
    voxel_size = 10
    locations = tiled_locations(voxel_size)
    nt.eq_(set(np.diff(locations, axis=0).flatten()), set((0, -400, 10)))


def test_build_voxel_synapse_count():
    distmap = [[(27, 1.5), (79, 3.4), (147, 7.8), (238, 1)]]
    voxel_size = 25
    voxel_volume = voxel_size**3
    count_per_slice = (np.array([1.5, 3.4, 7.8]) * voxel_volume).astype(int)
    result = build_voxel_synapse_count(distmap, voxel_size=voxel_size).raw
    npt.assert_equal(result[0][:10],
                     np.array([[0],
                               [count_per_slice[0]],
                               [count_per_slice[0]],
                               [count_per_slice[1]],
                               [count_per_slice[1]],
                               [count_per_slice[2]],
                               [count_per_slice[2]],
                               [count_per_slice[2]],
                               [count_per_slice[2]],
                               [0]]))


def test_find_cutoff_mean():
    synapses_per_connection = np.fromiter(chain([1] * 400,
                                                [2] * 800,
                                                [3] * 400,
                                                [4] * 200),
                                          dtype=int)
    value_count = pd.Series(synapses_per_connection).value_counts(sort=False)
    nt.assert_equal(find_cutoff_mean(value_count, 0), 1)
    nt.assert_equal(find_cutoff_mean(value_count, 0.8), 3)
    nt.assert_equal(find_cutoff_mean(value_count, 1), 4)


def test_prune_synapses_by_target_pathway():
    import matplotlib.pyplot as plt

    tgid = list(chain(list(range(100)) * 4,
                      list(range(100, 120)) * 8,
                      list(range(120, 140)) * 2))

    sgid = [1] * len(tgid)
    mtype = [1] * len(tgid)

    synapses = pd.DataFrame({'mtype': mtype, 'sgid': sgid, 'tgid': tgid})
    # print(len(synapses))
    plt.hist([len(prune_synapses_by_target_pathway(mtypes=[1],
                                                   synapses=synapses,
                                                   cutoff_var=1.0)) / float(len(synapses))
              for _ in range(100)])
    plt.show()
    # print(out)
    # print(len(out))
