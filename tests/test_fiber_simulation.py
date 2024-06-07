from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from bluepysnap.bbp import Cell
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from voxcell.nexus.voxelbrain import Atlas

import projectionizer.fiber_simulation as test_module

from utils import TEST_DATA_DIR

XZ = list("xz")
XYZ = list("xyz")
UVW = list("uvw")


def test__generate_kmeans_fibers():
    # coordinates for a square
    sqc = np.array(
        [
            [0, 0],
            [0, 2],
            [2, 2],
            [2, 0],
        ]
    )

    xv = np.array([1, 0])
    zv = np.array([0, 1])

    # create xz positions
    xz = np.concatenate(
        [
            sqc,
            sqc + 10 * xv,
            sqc + 10 * zv,
            sqc + 10,
        ]
    ).astype(float)

    cells = pd.DataFrame(xz, columns=XZ)
    v_dir = 1.0
    y_level = 0.0
    n_fibers = 4

    fibers = test_module._generate_kmeans_fibers(cells, n_fibers, v_dir, y_level)

    assert np.all(fibers["v"] == v_dir)
    assert np.all(fibers["y"] == y_level)
    assert len(fibers) == n_fibers

    assert_array_equal(
        fibers.sort_values(XZ)[XZ].values,  # pylint: disable=unsubscriptable-object
        [
            [1, 1],
            [1, 11],
            [11, 1],
            [11, 11],
        ],
    )


def test_generate_kmeans():
    # coordinates for a square
    sqc = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])

    xv = np.array([1, 0])
    zv = np.array([0, 1])

    # create xz positions of four squares
    xz = np.concatenate(
        [
            sqc,
            sqc + 10 * xv,
            sqc + 10 * zv,
            sqc + 10,
        ]
    ).astype(float)

    cells = pd.DataFrame(xz, columns=XZ)
    v_dir = 1.0
    y_level = 0.0
    n_fibers = 1
    bounding_rectangle = [[9, 9], [13, 13]]

    mock_cells = Mock(return_value=cells)
    node_population = Mock(get=mock_cells)

    fibers = test_module.generate_kmeans(
        node_population,
        n_fibers,
        v_dir,
        y_level,
        bounding_rectangle=bounding_rectangle,
    )

    assert np.all(fibers["v"] == v_dir)
    assert np.all(fibers["y"] == y_level)
    assert len(fibers) == n_fibers

    # only last 4 points are within bounding box, expect mean of those (n_fibers==1)
    assert_array_equal(fibers[XZ], [cells[-4:].mean()])

    region = "TEST_layers"
    mock_cells.reset_mock()
    fibers = test_module.generate_kmeans(node_population, n_fibers, v_dir, y_level, regions=region)

    # Check that cells.get was called with a query {Cell.REGION: region}
    assert {Cell.REGION: region} in mock_cells.call_args[0]

    # fibers should equal to mean of all cells (no bounding_rectangle, n_fibers=1)
    assert_array_equal(fibers[XZ], [cells.mean()])

    # Check that cells.get wasn't called with any args the second time
    fibers = test_module.generate_kmeans(node_population, n_fibers, v_dir, y_level)
    assert not mock_cells.call_args_list[1][0]


def test_get_fiber_directions():
    # rotations for the positions in orientation.nrrd are:
    # [0,...] : 90deg x-axis-wise
    # [1,...] : 90deg z-axis-wise
    # [2,...] : 180deg x-axis-wise
    fiber_pos = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    atlas = Atlas.open(str(TEST_DATA_DIR))

    res = test_module.get_fiber_directions(fiber_pos, atlas)

    assert_allclose(
        res,
        [
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
        ],
        atol=1e-15,
    )


def test_get_l5_l34_border_voxel_indices():
    atlas = Atlas.open(str(TEST_DATA_DIR))
    brain_regions = atlas.load_data("brain_regions")
    ret = test_module.get_l5_l34_border_voxel_indices(atlas, ["TEST_layers"])

    region_ids = brain_regions.raw[tuple(np.transpose(ret))]
    assert_array_equal([15], np.unique(region_ids))

    neighbors = np.array(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    def check_neighbors(ind, layer_ids):
        # check that any of the `layer_ids` is found in the neighbors
        idx = tuple(np.transpose(ind + neighbors))
        return np.any(np.in1d(brain_regions.raw[idx], layer_ids))

    assert np.all([check_neighbors(i, [14, 13]) for i in ret])


def test_mask_layers_in_regions():
    atlas = Atlas.open(str(TEST_DATA_DIR))
    brain_regions = atlas.load_data("brain_regions")

    mask = brain_regions.raw == 1121
    ret = test_module.mask_layers_in_regions(atlas, ["L3"], ["S1FL"])
    assert_array_equal(ret, mask)

    mask = brain_regions.raw == 1121
    mask |= brain_regions.raw == 1122
    mask |= brain_regions.raw == 1127
    mask |= brain_regions.raw == 1128
    ret = test_module.mask_layers_in_regions(atlas, ["L3", "L4"], ["S1FL", "S1HL"])
    assert_array_equal(ret, mask)


def test_mask_layer_6_bottom():
    atlas = Atlas.open(str(TEST_DATA_DIR))
    brain_regions = atlas.load_data("brain_regions")
    distance = atlas.load_data("[PH]y")

    # [PH]y is deliberately constructed so that the bottom of TEST_layers;L6 has a distance of 0
    mask = brain_regions.raw == 16
    mask &= distance.raw == 0

    ret = test_module.mask_layer_6_bottom(atlas, ["TEST_layers"])
    assert_array_equal(ret, mask)
    assert ret.sum() == 28 * 28


def test_ray_tracing():
    atlas = Atlas.open(str(TEST_DATA_DIR))
    brain_regions = atlas.load_data("brain_regions")
    distance = atlas.load_data("[PH]y")

    mask = brain_regions.raw == 16
    mask &= distance.raw == 0
    assert mask.sum() == 28 * 28

    ind_zero = np.transpose(np.where(mask))
    fiber_pos = distance.indices_to_positions(ind_zero) + np.array([0, 50, 0])
    fiber_dir = np.zeros(fiber_pos.shape) + np.array([0, 1, 0])
    fiber_dir[-1] = [0, 0, 1]  # change the last dir vector to miss the target
    ret = test_module.ray_tracing(atlas, mask, fiber_pos, fiber_dir)
    ind_fibers = distance.positions_to_indices(ret[list("xyz")].values)

    # Ray tracing should have found all but the last voxel indicated ind_zero
    assert len(ind_fibers) == len(ind_zero[:-1])
    assert_array_equal(ind_fibers, ind_zero[:-1])


def test_average_distance_to_nearest_neighbor():
    base = [1, 2, 3]
    pos = np.vstack(
        [
            np.tile(base, (1, 9)),
            np.tile(np.repeat(base, 3), 3),
            np.repeat(base, 9),
        ]
    ).T

    res = test_module.average_distance_to_nearest_neighbor(pos)

    assert res == 1


def test_get_orthonormal_basis_plane():
    for vector in [[1, 1, 1], [1, 1, 0], [1, 0, 0]]:
        basis_vectors = test_module.get_orthonormal_basis_plane(vector)
        x, y = basis_vectors.T
        assert_allclose(np.linalg.norm(basis_vectors, axis=0), 1)
        assert_almost_equal(0, np.dot(x, y))


def test_get_vectors_on_plane():
    fiber = [1, 2, 3, 4, 5, 6]
    distance = 10
    n_fibers = 20
    basis_vectors = test_module.get_orthonormal_basis_plane(fiber[-3:])
    vectors_3d = test_module.vectors_on_plane(basis_vectors, distance, n_fibers)
    assert len(vectors_3d) == n_fibers

    # The theoretical maximum distance (in case the plane was parallel to one of the axes
    # and the basis vectors were in 45 degree angle compared to that axis.
    max_distance = np.sqrt(2) * distance
    assert np.all(np.abs(vectors_3d) < max_distance)

    # To check that all of the vectors are on a same plane (orthogonal to direction vector)
    assert_allclose(np.dot(vectors_3d, fiber[-3:]), np.zeros(n_fibers), atol=1e-13)


def test_increase_fibers():
    fiber = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=list("xyzuvw"))
    dir_v = fiber[UVW].to_numpy()[0]
    n_fibers = 10

    new_fibers = test_module.increase_fibers(fiber, n_fibers)

    assert len(new_fibers) == n_fibers
    assert_array_equal(new_fibers[UVW].to_numpy(), np.tile(dir_v, (n_fibers, 1)))

    # to verify the fiber positions are on the same plane
    xyzs = new_fibers[XYZ].to_numpy()
    assert_allclose(np.matmul(xyzs, dir_v) - np.dot(fiber[XYZ], dir_v), 0, atol=1e-13)


@patch.object(test_module, "get_fiber_directions")
def test_generate_raycast(mock_get_fiber_directions):
    atlas = Atlas.open(str(TEST_DATA_DIR))
    n_voxels = 28 * 28
    n_fibers = 2 * n_voxels - 1
    mock_get_fiber_directions.return_value = np.tile(np.array((0, 1, 0)), (n_voxels, 1))

    fibers = test_module.generate_raycast(atlas, ["TEST"], n_fibers)
    assert len(fibers) == n_fibers
