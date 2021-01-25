'''Luigi task and functions for generating the fibers to a file.'''
import logging
from luigi import Config, Parameter, FloatParameter, IntParameter, BoolParameter, ListParameter
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from scipy.cluster.vq import kmeans

from bluepy.v2 import Circuit, Cell
import voxcell

from projectionizer.utils import mask_by_region, read_regions_from_manifest, XYZUVW

L = logging.getLogger(__name__)
XZ = list('xz')


class GenerateFibers(Config):  # pragma: no cover
    """Generate the fibers. """

    circuit_config = Parameter()
    use_kmeans = BoolParameter()
    n_fibers = IntParameter(default=np.inf)
    regions = ListParameter(default='')
    bounding_rectangle = ListParameter(default=[])
    out_file = Parameter()
    v_direction = FloatParameter(default=1.0)
    y_level = FloatParameter(default=0.)

    def run(self):
        circuit = Circuit(self.circuit_config)
        regions = self.regions
        if self.use_kmeans:
            fibers = generate_kmeans(circuit,
                                     self.n_fibers,
                                     self.v_direction,
                                     self.y_level,
                                     regions=regions,
                                     bounding_rectangle=self.bounding_rectangle)
        else:
            if not regions:
                regions = read_regions_from_manifest(self.circuit_config)
                assert regions, 'No regions defined'

            fibers = generate_raycast(circuit.atlas, regions)

        L.info('Saving fibers to %s', self.out_file)
        fibers.to_csv(self.out_file, index=False, sep=",")


# -- K means clustering --

def generate_kmeans(circuit, n_fibers, v_dir, y_level, regions='', bounding_rectangle=''):
    '''Generate fibers using k-means clustering.'''
    assert np.isfinite(n_fibers), 'Number of fibers to generate not given'

    if bounding_rectangle:
        min_xz, max_xz = bounding_rectangle
        cells = circuit.cells.get(properties=XZ)
        cells = cells[(min_xz[0] < cells.x) &
                      (cells.x < max_xz[1]) &
                      (min_xz[1] < cells.z) &
                      (cells.z < max_xz[1])].reset_index(drop=True)
    elif regions:
        cells = circuit.cells.get({Cell.REGION: regions}, properties=XZ)
    else:
        cells = circuit.cells.get(properties=XZ)

    return _generate_kmeans_fibers(cells, n_fibers, v_dir, y_level)


def _generate_kmeans_fibers(cells, n_fibers, v_dir, y_level):
    '''Generate the fibers dataframe using k-means clustering.'''
    codebook, _ = kmeans(cells[XZ].values, n_fibers)

    fiber_pos = pd.DataFrame(codebook, columns=XZ)
    fiber_pos["v"] = v_dir
    fiber_pos["y"] = y_level
    fiber_pos["u"] = 0.0
    fiber_pos["w"] = 0.0

    return fiber_pos


# -- Ray casting --

def generate_raycast(atlas, regions):
    '''Generate fibers using the ray casting.

    Tracing back from L5/L43 boundary along the orientations to bottom of L6 and picking
    those fibers that do hit the bottom of L6 voxels.

    Args:
        atlas (voxcell.Atlas): atlas instance for the circuit
        regions (list): list of region acronyms

    Return:
        (pandas.DataFrame): virtual fibers found
    '''
    distance = atlas.load_data('[PH]y')
    mask = mask_layer_6_bottom(atlas, regions)
    fiber_pos = distance.indices_to_positions(get_l5_l34_border_voxel_indices(atlas, regions))
    fiber_pos += distance.voxel_dimensions / 2.

    count = None  # should be a parameter
    if count is not None:
        fiber_pos = fiber_pos[np.random.choice(np.arange(len(fiber_pos)), count, replace=False)]

    fiber_dir = get_fiber_directions(fiber_pos, atlas)

    return ray_tracing(atlas, mask, fiber_pos, fiber_dir)


def ray_tracing(atlas, target_mask, fiber_positions, fiber_directions):
    '''Get virtual fiber start positions by ray_tracing.

    Args:
        atlas (voxcell.Atlas): atlas instance for the circuit
        target_mask (numpy.array): 3D array masking the potential startpoints (e.g. bottom of L4)
        fiber_positions (numpy.array): fiber positions to trace back from (e.g., L4/L5 boundary
                                       voxel positions)
        fiber_directions (numpy.array): directions of the fibers at the positions given at fiber_pos

    Return:
        (pandas.DataFrame): virtual fibers found
    '''
    ret = []
    distance = atlas.load_data('[PH]y')

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
    '''Get the mask for the bottom of layer6.'''
    distance = atlas.load_data('[PH]y')
    mask6 = mask_layers_in_regions(atlas, ['L6'], regions)
    distance.raw[np.invert(mask6)] = np.nan
    min_dist = np.min(distance.raw[np.isfinite(distance.raw)])

    return distance.raw == min_dist


def get_l5_l34_border_voxel_indices(atlas, regions):
    '''Get the fiber indices.

    i.e., the indices of the voxels that lie on the border of L5 and L3/L4.
    '''
    mask_l5 = mask_layers_in_regions(atlas, ['L5'], regions)
    mask_l34 = mask_layers_in_regions(atlas, ['L3', 'L4'], regions)
    mask = mask_l5 & nd.binary_dilation(mask_l34)

    return np.transpose(np.where(mask))


def get_fiber_directions(fiber_positions, atlas):
    '''Get the fiber directions at positions defined in fiber_positions.'''
    orientation = atlas.load_data('orientation', cls=voxcell.OrientationField)
    orientation.raw = orientation.raw.astype(np.int8)
    y_vec = np.array([0, 1, 0])
    R = orientation.lookup(fiber_positions)

    return np.matmul(R, y_vec)


def mask_layers_in_regions(atlas, layers, regions):
    '''Get the mask for defined layers in all defined regions.'''
    ids = get_region_ids(atlas, layers, regions)

    return mask_by_region(ids, atlas)


def get_region_ids(atlas, layers, regions):
    '''Get region id's for the regions and layers.'''
    rmap = atlas.load_region_map()
    regex_str_regions = '@^({})$'.format('|'.join(regions))
    regex_str_layers = '@^.*({})$'.format('|'.join(layers))

    id_regions_children = rmap.find(regex_str_regions, attr='acronym', with_descendants=True)
    id_layers_all_regions = rmap.find(regex_str_layers, attr='acronym')
    id_wanted_layers = set.intersection(id_regions_children, id_layers_all_regions)

    return list(id_wanted_layers)
