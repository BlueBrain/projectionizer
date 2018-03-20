import numpy as np
import pandas as pd
from nose.tools import ok_
from numpy.testing import assert_allclose, assert_equal

from projectionizer.sscx import *
from projectionizer.step_1_assign import assign_synapse_fiber
from projectionizer.utils import choice

from mocks import (
    create_candidates,
    create_synapse_counts,
    create_virtual_fibers
)


def test_assign_synapse_fiber():
    np.random.seed(37)
    candidates = create_candidates()
    virtual_fibers = create_virtual_fibers()
    ret = assign_synapse_fiber(candidates,
                               virtual_fibers,
                               sigma=1)
    ok_(ret.equals(pd.DataFrame({'sgid': [0, 2, 1]})))


def test_choice():
    np.random.seed(0)
    indices = choice(np.array([[1., 2, 3, 4],
                               [0, 0, 1, 0],
                               [6, 5, 4, 0]]))
    assert_equal(indices,
                 [2, 2, 1])


def test_recipe_to_height_and_density():
    profile = [[0.05, 0.01], [0.15, 0.02], [0.25, 0.03], [0.35, 0.04], [0.45, 0.04],
               [0.55, 0.04], [0.65, 0.03], [0.75, 0.02], [0.85, 0.01], [0.95, 0.01]]
    assert_allclose(recipe_to_height_and_density(4, 0, 3, 0.5,
                                                 profile),
                    [(1225.43431672, 0.01),
                     (1262.0377547759999, 0.02),
                     (1298.6411928319999, 0.029999999999999999),
                     (1335.2446308880001, 0.040000000000000001),
                     (1371.848068944, 0.040000000000000001),
                     (1408.451507, 0.040000000000000001),
                     (1445.054945056, 0.029999999999999999),
                     (1481.6583831119999, 0.02),
                     (1518.2618211680001, 0.01),
                     (1554.8652592240001, 0.01),
                     (1591.46869728, 0.01)])
