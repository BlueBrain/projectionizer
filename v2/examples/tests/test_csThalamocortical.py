import logging
from itertools import chain, repeat

import nose.tools as nt
import numpy as np
import numpy.testing as npt
import pandas as pd
from mock import MagicMock
from nose.tools import eq_, ok_, raises
from numpy.linalg import norm

from examples.mini_col_locations import tiled_locations
from examples.SSCX_Thalamocortical_VPM_hex import (_find_cutoff_means,
                                                   assign_synapse_virtual_fibers,
                                                   build_voxel_synapse_count,
                                                   choice,
                                                   find_cutoff_mean_per_mtype,
                                                   first_partition,
                                                   mask_far_fibers,
                                                   prune_synapses_by_target_pathway)

L = logging.getLogger(__name__)

np.random.seed(0)


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


def test_find_cutoff_means_per_mtype():
    synapses_per_connection = np.fromiter(chain([1] * 400,
                                                [2] * 800,
                                                [3] * 400,
                                                [4] * 200),
                                          dtype=int)
    value_count = pd.Series(synapses_per_connection).value_counts(sort=False)
    nt.assert_equal(find_cutoff_mean_per_mtype(value_count, 0), 1)
    nt.assert_equal(find_cutoff_mean_per_mtype(value_count, 0.8), 3)
    nt.assert_equal(find_cutoff_mean_per_mtype(value_count, 1), 4)


def test_prune_synapses_by_target_pathway():

    # Flat distribution #synapses/connection
    tgid = list(chain.from_iterable([[i] * i for i in range(100)]))
    sgid = [1] * len(tgid)
    mtype = [1] * len(tgid)

    synapses = pd.DataFrame({'mtype': mtype, 'sgid': sgid, 'tgid': tgid})

    pruned_fraction = [len(prune_synapses_by_target_pathway(synapses=synapses,
                                                            synaptical_fraction=0.5,
                                                            cutoff_var=1.0)) / float(len(synapses))
                       for _ in range(100)]
    npt.assert_almost_equal(np.mean(pruned_fraction), 0.5, decimal=2)

    # import matplotlib.pyplot as plt
    # plt.hist(pruned_fraction)
    # plt.show()


def test_first_partition():
    array = np.array([[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 10],
                      [15, 14, 13, 12, 11],
                      [12, 12, 12, 12, 12],
                      [17, 18, 19, 20, 21]])
    partition, indices = first_partition(array, 3)
    npt.assert_equal(partition,
                     np.array([[2,  1,  3],
                               [7,  6,  8],
                               [12, 11, 13],
                               [12, 12, 12],
                               [18, 17, 19]]))

    npt.assert_equal(indices,
                     np.array([[1, 0, 2],
                               [1, 0, 2],
                               [3, 4, 2],
                               [3, 2, 0],
                               [1, 0, 2]]))


def test_mask_far_fibers():
    fibers = np.array([[[0, 0], [2, 2], [3, 3], [4, 4]],
                       [[-1, 0], [2.5, 2.0], [3, 3], [4, 2]]])
    mask = mask_far_fibers(fibers, origin=[2, 2], exclusion_box=(2, 2))
    npt.assert_equal(mask,
                     [[False,  True, True, False],
                      [False,  True,  True, False]])


def test_choice():
    np.random.seed(0)

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    X = np.arange(20)
    Ys = np.array([gaussian(X, mu=5, sig=1) for _ in range(100)])
    indices = choice(Ys)
    npt.assert_almost_equal(np.mean(X[indices]), 5, decimal=1)
    npt.assert_almost_equal(np.var(X[indices]), 1, decimal=1)


def test_assign_synapse():
    np.random.seed(0)
    low = 0
    high = 10
    n_synapses = 1000
    n_fibers = 1000
    x = np.random.uniform(low=low, high=high, size=n_synapses)
    z = np.random.uniform(low=low, high=high, size=n_synapses)
    synapses = pd.DataFrame({'x': x, 'z': z})

    x = np.random.uniform(low=low, high=high, size=n_fibers)
    z = np.random.uniform(low=low, high=high, size=n_fibers)
    fibers = np.array(zip(x, z))

    syn = assign_synapse_virtual_fibers(synapses, fibers)
    distance = norm(syn[['x', 'z']] - fibers[syn.sgid], axis=1)
    nt.ok_(np.mean(distance) < 1)
