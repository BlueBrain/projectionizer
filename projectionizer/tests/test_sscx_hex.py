import os
from voxcell import VoxelData
import numpy as np
from nose.tools import eq_, ok_, raises
from projectionizer import sscx_hex


HEX_EDGE_LEN = 230.92
TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
LOCATIONS_PATH = os.path.join(TEMPLATES, 'rat_fibers.csv')


def test_tiled_locations():
    voxel_size = 10
    locations = sscx_hex.tiled_locations(voxel_size,
                                         hex_edge_len=HEX_EDGE_LEN,
                                         locations_path=LOCATIONS_PATH)
    eq_(set(np.diff(locations, axis=0).flatten()), set((0, -400, 10)))


def test_hexagon():
    points = sscx_hex.hexagon(hex_edge_len=HEX_EDGE_LEN)
    eq_(len(points), 7)


def test_get_virtual_fiber_locations(apron_size=0.0):
    vf = sscx_hex.get_virtual_fiber_locations(hex_edge_len=HEX_EDGE_LEN,
                                              locations_path=LOCATIONS_PATH,
                                              apron_size=0.0)
    eq_(len(vf), 407)

    vf = sscx_hex.get_virtual_fiber_locations(hex_edge_len=HEX_EDGE_LEN,
                                              locations_path=LOCATIONS_PATH,
                                              apron_size=10)
    ok_(len(vf) > 407)


def test_voxel_space():
    MAX_HEIGHT = 2081.75641787
    vs = sscx_hex.voxel_space(hex_edge_len=HEX_EDGE_LEN,
                              locations_path=LOCATIONS_PATH,
                              max_height=MAX_HEIGHT,
                              voxel_size_um=10)
    ok_(isinstance(vs, VoxelData))
    eq_(vs.raw.shape, (46, 208, 40))

    vs = sscx_hex.voxel_space(hex_edge_len=HEX_EDGE_LEN,
                              locations_path=LOCATIONS_PATH,
                              max_height=MAX_HEIGHT,
                              voxel_size_um=100)
    eq_(vs.raw.shape, (5, 20, 4))


def test_get_minicol_virtual_fibers():
    df = sscx_hex.get_minicol_virtual_fibers(apron_size=0.0,
                                             hex_edge_len=HEX_EDGE_LEN,
                                             locations_path=LOCATIONS_PATH)
    eq_(len(df), 407)

    df = sscx_hex.get_minicol_virtual_fibers(apron_size=10.0,
                                             hex_edge_len=HEX_EDGE_LEN,
                                             locations_path=LOCATIONS_PATH)
    ok_(len(df) > 407)
    ok_(len(df.index.unique()) > 407)
