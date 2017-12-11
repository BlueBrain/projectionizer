#from numpy.testing import assert_allclose, assert_equal
#from nose.tools import ok_, eq_, raises
#
#from bluepy.v2.enums import Section
#from neurom import NeuriteType
#
#import pandas as pd
#import numpy as np
#
#from projectionizer import utils
#
#
#def test_normalize_probability():
#    p = np.array([1, 0])
#    ret = utils.normalize_probability(p)
#    assert_equal(p, ret)
#
#
#def test_segment_pref():
#    df = pd.DataFrame({Section.NEURITE_TYPE: [NeuriteType.axon,
#                                              NeuriteType.axon,
#                                              NeuriteType.basal_dendrite,
#                                              NeuriteType.apical_dendrite,
#                                              ]
#                       })
#    ret = utils.segment_pref(df)
#    ok_(isinstance(ret, pd.Series))
#    assert_equal(ret.values, np.array([0., 0., 1., 1.]))
#
#
#def test_in_bounding_box():
#    min_xyz = np.array([0, 0, 0], dtype=float)
#    max_xyz = np.array([10, 10, 10], dtype=float)
#
#    for axis in ('x', 'y', 'z'):
#        df = pd.DataFrame({'x': np.arange(1, 2),
#                           'y': np.arange(1, 2),
#                           'z': np.arange(1, 2)})
#        ret = utils.in_bounding_box(min_xyz, max_xyz, df)
#        assert_equal(ret.values, [True, ])
#
#        #check for violation of min_xyz
#        df[axis].iloc[0] = 0
#        ret = utils.in_bounding_box(min_xyz, max_xyz, df)
#        assert_equal(ret.values, [False, ])
#        df[axis].iloc[0] = 1
#
#        #check for violation of max_xyz
#        df[axis].iloc[0] = 10
#        ret = utils.in_bounding_box(min_xyz, max_xyz, df)
#        assert_equal(ret.values, [False, ])
#        df[axis].iloc[0] = 1
#
#    df = pd.DataFrame({'x': np.arange(0, 10),
#                       'y': np.arange(5, 15),
#                       'z': np.arange(5, 15)})
#    ret = utils.in_bounding_box(min_xyz, max_xyz, df)
#    assert_equal(ret.values, [False, # x == 0, fails
#                              True, True, True, True,
#                              False, False, False, False, False, # y/z > 9
#                              ])
