"""Luigi task and functions for generating the fibers to a file."""
import logging

import numpy as np
import pandas as pd
import scipy.ndimage as nd
import voxcell
from bluepy import Cell, Circuit
from luigi import Config, FloatParameter, IntParameter, ListParameter, PathParameter
from scipy.cluster.vq import kmeans
from scipy.spatial import cKDTree

from projectionizer.utils import XYZUVW, mask_by_region, read_regions_from_manifest

L = logging.getLogger(__name__)
XZ = list("xz")
XYZ = list("xyz")
UVW = list("uvw")
# Fallback distance (in microns) in case the average distance can't be calculated
FALLBACK_AVG_DISTANCE = 10


class GenerateFibers(Config):  # pragma: no cover
    """Generate the fibers."""

    circuit_config = PathParameter()
    n_fibers = IntParameter(default=np.inf)
    regions = ListParameter(default="")
    out_file = PathParameter()

    def _save_as_csv(self, fibers):
        L.info("Saving fibers to %s", self.out_file)
        fibers.to_csv(self.out_file, index=False, sep=",")

    def _generate_fibers(self):
        circuit = Circuit(self.circuit_config)
        regions = self.regions

        if not regions:
            regions = read_regions_from_manifest(self.circuit_config)
            assert regions, "No regions defined"

        return generate_raycast(circuit.atlas, regions, self.n_fibers)

    def run(self):
        fibers = self._generate_fibers()
        self._save_as_csv(fibers)


class GenerateFibersHex(GenerateFibers):  # pragma: no cover
    """Generate the fibers for a column."""

    bounding_rectangle = ListParameter(default=[])
    v_direction = FloatParameter(default=1.0)
    y_level = FloatParameter(default=0.0)

    def _generate_fibers(self):
        return generate_kmeans(
            Circuit(self.circuit_config),
            self.n_fibers,
            self.v_direction,
            self.y_level,
            regions=self.regions,
            bounding_rectangle=self.bounding_rectangle,
        )


# -- K means clustering --


def generate_kmeans(circuit, n_fibers, v_dir, y_level, regions="", bounding_rectangle=""):
    """Generate fibers using k-means clustering."""
    assert np.isfinite(n_fibers), "Number of fibers to generate not given"

    if bounding_rectangle:
        min_xz, max_xz = bounding_rectangle
        cells = circuit.cells.get(properties=XZ)
        cells = cells[
            (min_xz[0] < cells.x)
            & (cells.x < max_xz[1])
            & (min_xz[1] < cells.z)
            & (cells.z < max_xz[1])
        ].reset_index(drop=True)
    elif regions:
        cells = circuit.cells.get({Cell.REGION: regions}, properties=XZ)
    else:
        cells = circuit.cells.get(properties=XZ)

    return _generate_kmeans_fibers(cells, n_fibers, v_dir, y_level)


def _generate_kmeans_fibers(cells, n_fibers, v_dir, y_level):
    """Generate the fibers dataframe using k-means clustering."""
    codebook, _ = kmeans(cells[XZ].values, n_fibers)

    fiber_pos = pd.DataFrame(codebook, columns=XZ)
    fiber_pos["v"] = v_dir
    fiber_pos["y"] = y_level
    fiber_pos["u"] = 0.0
    fiber_pos["w"] = 0.0

    return fiber_pos[XYZUVW]


# -- Ray casting --


def generate_raycast(atlas, regions, n_fibers):
    """Generate fibers using the ray casting.

    Tracing back from L5/L43 boundary along the orientations to bottom of L6 and picking
    those fibers that do hit the bottom of L6 voxels.

    Args:
        atlas (voxcell.Atlas): atlas instance for the circuit
        regions (list): list of region acronyms

    Return:
        (pandas.DataFrame): virtual fibers found
    """
    distance = atlas.load_data("[PH]y")
    mask = mask_layer_6_bottom(atlas, regions)
    fiber_pos = distance.indices_to_positions(get_l5_l34_border_voxel_indices(atlas, regions))
    fiber_pos += distance.voxel_dimensions / 2.0

    fiber_dir = get_fiber_directions(fiber_pos, atlas)
    fibers = ray_tracing(atlas, mask, fiber_pos, fiber_dir)

    if np.isfinite(n_fibers):
        if n_fibers > len(fibers):
            fibers = increase_fibers(fibers, n_fibers)
        if n_fibers < len(fibers):
            picked = np.random.choice(np.arange(len(fibers)), n_fibers, replace=False)
            fibers = fibers.iloc[picked].reset_index(drop=True)

        assert n_fibers == len(fibers)

    return fibers


def ray_tracing(atlas, target_mask, fiber_positions, fiber_directions):
    """Get virtual fiber start positions by ray_tracing.

    Args:
        atlas (voxcell.Atlas): atlas instance for the circuit
        target_mask (np.array): 3D array masking the potential start points (e.g. bottom of L4)
        fiber_positions (np.array): fiber positions to trace back from (e.g., L4/L5 boundary
                                       voxel positions)
        fiber_directions (np.array): directions of the fibers at the positions given at fiber_pos

    Return:
        (pandas.DataFrame): virtual fibers found
    """
    ret = []
    distance = atlas.load_data("[PH]y")

    # TODO: more effective way of tracing the fibers to bottom of L6.
    for pos, dirs in zip(fiber_positions, fiber_directions):
        try:
            while not target_mask[tuple(distance.positions_to_indices(pos))]:
                pos -= dirs * distance.voxel_dimensions[1]
        except voxcell.exceptions.VoxcellError:
            # Expecting out of bounds error if ray tracing did not hit any voxels in mask
            continue

        ret.append(np.concatenate((pos, dirs)))

    ret = np.vstack(ret)

    return pd.DataFrame(ret, columns=XYZUVW)


def mask_layer_6_bottom(atlas, regions):
    """Get the mask for the bottom of layer6."""
    distance = atlas.load_data("[PH]y")
    mask6 = mask_layers_in_regions(atlas, ["L6"], regions)
    distance.raw[np.invert(mask6)] = np.nan
    min_dist = np.min(distance.raw[np.isfinite(distance.raw)])

    return distance.raw == min_dist


def get_l5_l34_border_voxel_indices(atlas, regions):
    """Get the fiber indices.

    i.e., the indices of the voxels that lie on the border of L5 and L3/L4.
    """
    mask_l5 = mask_layers_in_regions(atlas, ["L5"], regions)
    mask_l34 = mask_layers_in_regions(atlas, ["L3", "L4"], regions)
    mask = mask_l5 & nd.binary_dilation(mask_l34)

    return np.transpose(np.where(mask))


def get_fiber_directions(fiber_positions, atlas):
    """Get the fiber directions at positions defined in fiber_positions."""
    orientation = atlas.load_data("orientation", cls=voxcell.OrientationField)
    orientation.raw = orientation.raw.astype(np.int8)
    y_vec = np.array([0, 1, 0])
    R = orientation.lookup(fiber_positions)

    return np.matmul(R, y_vec)


def mask_layers_in_regions(atlas, layers, regions):
    """Get the mask for defined layers in all defined regions."""
    ids = get_region_ids(atlas, layers, regions)

    return mask_by_region(ids, atlas)


def get_region_ids(atlas, layers, regions):
    """Get region id's for the regions and layers."""
    rmap = atlas.load_region_map()
    regex_str_regions = f"@^({'|'.join(regions)})$"
    regex_str_layers = f"@^.*({'|'.join(layers)})$"

    id_regions_children = rmap.find(regex_str_regions, attr="acronym", with_descendants=True)
    id_layers_all_regions = rmap.find(regex_str_layers, attr="acronym")
    id_wanted_layers = set.intersection(id_regions_children, id_layers_all_regions)

    return list(id_wanted_layers)


def average_distance_to_nearest_neighbor(xyzs):
    """Average distance to nearest neighbor for all samples."""
    tree = cKDTree(xyzs)
    distances, _ = tree.query(xyzs, 2)
    return distances[:, 1].mean()


def get_orthonormal_basis_plane(vector):
    """Get orthonormal basis vectors for a plane orthogonal to the given vector.

    Function assumes the vector is normalized.
    """
    # plane passing through origin given its normal vector [a, b, c]: aX + bY + cZ = 0,
    # NOTE: Origin assumed for simplicity. We are only interested in the base vectors.
    a, b, c = vector

    # Computing basis vectors for the plane. Solving the plane equation for Z
    # Z = -(aX + bY) / c
    # [X, Y, Z] = [X, Y, -(aX+bY)/c] = X[1, 0, -a/c] + Y[0, 1, -b/c]
    # constructing two linearly independent vectors by letting
    #   [X, Y] = [1, 0] => [X, Y, Z] = 1[1, 0 -a/c] + 0[0, 1, -b/c] = [1, 0, -a/c]
    #   [X, Y] = [0, 1] => [X, Y, Z] = 0[1, 0 -a/c] + 1[0, 1, -b/c] = [0, 1, -b/c]
    if np.abs(c) > 0.05:
        u = [1, 0, -a / c]
        v = [0, 1, -b / c]
    elif np.abs(b) > 0.05:
        # if c close to zero, just solve for Y and do the same
        u = [1, -a / b, 0]
        v = [0, -c / b, 1]
    else:
        # if both b,c close to zero solve for X
        u = [-b / a, 1, 0]
        v = [-c / a, 0, 1]

    basis_vectors = np.array([u, v]).T

    # columns of Q of the QR decomposition are the orthonormalized basis vectors that span the plane
    return np.linalg.qr(basis_vectors)[0]


def vectors_on_plane(basis_vectors, avg_distance, n_fibers):
    """Return position vectors on a plane spanned by basis vectors.

    Length of each vector is 1.0 * distance in direction of either basis vector at maximum."""
    vectors_2d = np.random.uniform(-1, 1, [n_fibers, 2]) * avg_distance
    return np.matmul(vectors_2d, basis_vectors.T)


def increase_fibers(fibers, n_fibers):
    """Increase the fiber count so that it is at least n_fibers.

    Returns a DataFrame of bunches of vectors around the original fiber (not included).
    """
    ratio = np.ceil(n_fibers / len(fibers)).astype(int)
    distance = average_distance_to_nearest_neighbor(fibers[XYZ].to_numpy())

    if not np.isfinite(distance):  # Happens e.g., when len(fibers)==1
        distance = FALLBACK_AVG_DISTANCE

    new_fibers = []
    for _, fiber in fibers.iterrows():
        start_xyz = fiber[XYZ].to_numpy()
        dir_v = fiber[UVW].to_numpy()
        basis = get_orthonormal_basis_plane(dir_v)
        new_xyz = start_xyz + vectors_on_plane(basis, distance, ratio)
        new_fibers.append(np.hstack((new_xyz, np.tile(dir_v, (ratio, 1)))))

    return pd.DataFrame(np.vstack(new_fibers), columns=XYZUVW)
