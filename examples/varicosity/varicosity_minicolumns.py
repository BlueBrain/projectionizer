#!/usr/bin/env python
'''Test run of Projectionizer code for NSETM-206

The densities and are very low, which results in having to oversample quite a bit

Example config.yaml:

name: CHAT
density:
    # CHAT - Varicosities / 1000 um^3
    mean:
        '1': 0.4799547285
        ...
        '6': 0.3252380622

circuit_config: /path/to/a/config/CircuitConfig
hex_edge_len: 230.92  # the length of a hexagon side
hex_apron_size: 50 # include fibers from outside of the central column
                   # by a bounding box increased by this size
voxel_size_um: 50  #
oversampling: 20.0 # number of times to increase the density
fiber_locations: rat_fibers.csv # path to the csv containing fiber locations
sigma: 50  # parameter for gaussian weighting of which fiber to pick relative to synapse
closest_count: 25  # how many fibers to consider when choosing the source gid for a synapse
layers:  #layer heights
    - ['6', 700.37845971]
    ...
    - ['1', 164.94915873]

synapse_parameters: # same as in general Projectionizer config
'''
import argparse
import logging
import math
import os
import shutil

import numpy as np
import pandas as pd

from voxcell import VoxelData

from projectionizer import sscx_hex, luigi_utils, synapses, straight_fibers, utils, write_nrn
import yaml


L = logging.getLogger(__name__)


def build_voxel_synapse_count(hex_edge_len,
                              fiber_locations_path,
                              layers,
                              synapse_density,
                              oversampling,
                              voxel_size_um=10):
    '''create the VoxelData set with the number of desired synapses per voxel'''
    max_height = sum(h for _, h in layers)
    voxels = sscx_hex.voxel_space(hex_edge_len=hex_edge_len,
                                  locations_path=fiber_locations_path,
                                  max_height=max_height,
                                  voxel_size_um=voxel_size_um
                                  )

    xyz = voxels.indices_to_positions(np.indices(
        voxels.raw.shape).transpose(1, 2, 3, 0) + (0.5, 0.5, 0.5))
    height = voxels.with_data(xyz[:, :, :, 1])

    res = synapses.build_synapses_default(
        height, synapse_density, oversampling)
    return res


def sample_synapses(circuit_path, voxel_synapse_count):
    '''sample the circuit based on the number of synapses wanted

    Args:
        circuit_path(path str): path to directory containing CircuitConfig
        voxel_synapse_count(VoxelData): each voxel contains the count of synapses
        that should be sample from the circuit

    Returns:
        synapse DataFrame
    '''
    syns = synapses.pick_synapses(circuit_path, voxel_synapse_count, None)
    syns.reset_index(inplace=True, drop=True)
    syns.columns = map(str, syns.columns)
    remove_cols = [u'Segment.X1', u'Segment.Y1', u'Segment.Z1',
                   u'Segment.X2', u'Segment.Y2', u'Segment.Z2']
    syns.drop(remove_cols, axis=1, inplace=True)

    # set datatypes so less memory
    syns['section_id'] = syns['Section.ID'].values.astype(np.int32)
    syns['segment_id'] = syns['Segment.ID'].values.astype(np.int32)
    syns['tgid'] = syns['gid'].values.astype(np.int32)
    syns['sgid_path_distance'] = syns['segment_length'].values.astype(np.float32)

    remove_cols = [u'Section.ID', u'Segment.ID', u'gid', 'segment_length', ]
    syns.drop(remove_cols, axis=1, inplace=True)

    return syns


def get_synapse_parameters(synapse_parameters):
    '''populate the parameters for the synapses'''
    def get_gamma_parameters(name):
        '''transform mean/sigma parameters as per original projectionizer code'''
        mn = synapse_parameters[name + '_mean']
        sd = synapse_parameters[name + '_sigma']
        return ((mn / sd) ** 2, (sd ** 2) / mn)  # k, theta or shape, scale

    params = {'id': synapse_parameters['synapse_type'],
              'gsyn': get_gamma_parameters('gsyn'),
              'Use': get_gamma_parameters('use'),
              'D': get_gamma_parameters('D'),
              'F': get_gamma_parameters('F'),
              'DTC': get_gamma_parameters('DTC'),
              'ASE': get_gamma_parameters('ASE'),
              }
    return params


def write_output(syns, synapse_parameters, virtual_fibers, sgid_offset, output):
    '''Create nrn files, targets, and fibers'''
    # write virtual-fibers
    virtual_fibers.index += sgid_offset
    virtual_fibers.to_csv(os.path.join(output, 'virtual-fibers.csv'), index_label='sgid')

    syns['location'] = 1.
    syns['sgid'] += sgid_offset
    synapses.organize_indices(syns)

    # write proj_nrn_*.h5
    path = os.path.join(output, 'proj_nrn.h5')
    itr = syns.groupby('tgid')
    write_nrn.write_synapses(path, itr, synapse_parameters, efferent=False)

    path = os.path.join(output, 'proj_nrn_efferent.h5')
    itr = syns.groupby('sgid')
    write_nrn.write_synapses(path, itr, synapse_parameters, efferent=True)

    # write summary
    path = os.path.join(output, 'proj_nrn_summary.h5')
    write_nrn.write_synapses_summary(path=path, synapses=syns)

    # write target file
    path = os.path.join(output, 'user.target')
    write_nrn.write_user_target(path, syns, name='proj_Thalamocortical_VPM_Source')


def assign_synapses(voxel_synapse_count, virtual_fibers, synapses_xyz, closest_count, sigma):
    '''based on the sample synapse locations, assign them to a source fiber

    Args:
        voxel_synapse_count(VoxelData): number of desired synapses per voxel
        virtual_fibers(dataframe of virtual fiber locations):
        synapses_xyz(dataframe): x/y/z positions of fibers
        closest_count(int): fibers count to consider when choosing the source gid for a synapse
        sigma(float): parameter for gaussian weighting of which fiber to pick relative to synapse
    '''
    closest_fibers_per_vox = straight_fibers.closest_fibers_per_voxel(
        voxel_synapse_count, virtual_fibers, closest_count)
    synapses_indices = pd.DataFrame(voxel_synapse_count.positions_to_indices(synapses_xyz.values),
                                    columns=utils.IJK)
    candidates = straight_fibers.candidate_fibers_per_synapse(
        synapses_xyz, synapses_indices, closest_fibers_per_vox)
    sgids = straight_fibers.assign_synapse_fiber(candidates, virtual_fibers, sigma).values
    return sgids


def get_voxel_synapse_count(config, fibers_path):
    ''''''
    means = config['density']['mean']
    synapse_density = []
    start = 0.
    for name, height in config['layers']:
        synapse_density.append([start, means[name] / 1000.])
        start += height
    synapse_density.append([start, means[name] / 1000.])
    synapse_density = [synapse_density, ]

    voxel_synapse_count = build_voxel_synapse_count(
        config['hex_edge_len'],
        fibers_path,
        config['layers'],
        synapse_density,
        config['oversampling'],
        config['voxel_size_um'])

    return voxel_synapse_count


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output',
                        help='Output directory')
    parser.add_argument('-c', '--config',
                        help='Config file')
    parser.add_argument('-g', '--graphs', action='store_true', default=False,
                        help='Create graphs')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='-v for INFO, -vv for DEBUG')

    return parser


def main(args):
    logging.basicConfig(level=(logging.WARNING,
                               logging.INFO,
                               logging.DEBUG)[min(args.verbose, 2)])

    with open(args.config) as fd:
        config = yaml.load(fd)

    fibers_path = luigi_utils.CommonParams.load_data(config['fiber_locations'])
    voxel_synapse_count = get_voxel_synapse_count(config, fibers_path)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    shutil.copy(args.config, args.output)

    sample_path = os.path.join(args.output, 'sample.feather')
    if os.path.exists(sample_path):
        L.info('Loading samples from: %s', sample_path)
        syns = utils.read_feather(sample_path)
    else:
        L.info('Performing sampling, saving to: %s', sample_path)
        circuit_path = os.path.dirname(config['circuit_config'])
        syns = sample_synapses(circuit_path, voxel_synapse_count)
        utils.write_feather(sample_path, syns)

    virtual_fibers = sscx_hex.get_minicol_virtual_fibers(apron_size=config['hex_apron_size'],
                                                         hex_edge_len=config['hex_edge_len'],
                                                         locations_path=fibers_path)
    syns_path = os.path.join(args.output, 'synapses.feather')
    if os.path.exists(syns_path):
        L.info('Loading synapses from: %s', syns_path)
        syns = utils.load(syns_path)
    else:
        L.info('Assigning synapses, saving to: %s', syns_path)
        syns = syns.sample(frac=1. / config['oversampling'])
        syns['sgid'] = assign_synapses(
            voxel_synapse_count, virtual_fibers, syns[utils.XYZ],
            config['closest_count'], config['sigma'])
        utils.write_feather(syns_path, syns)

    # write raw synapses
    synapse_parameters = get_synapse_parameters(config['synapse_parameters'])
    write_output(syns,
                 synapse_parameters,
                 virtual_fibers,
                 sgid_offset=config['sgid_offset'],
                 output=args.output)

    if args.graphs:
        import analysis
        analysis.plot_density(syns, args.output, config['name'], config['layers'])
        analysis.plot_synapses_per_connection(syns, args.output, config['name'])


if __name__ == '__main__':
    PARSER = get_parser()
    main(PARSER.parse_args())

#def test_build_voxel_synapse_count():
#    distmap = [[(27, 1.5), (79, 3.4), (147, 7.8), (238, 1)]]
#    voxel_size = 25
#    voxel_volume = voxel_size**3
#    count_per_slice = (np.array([1.5, 3.4, 7.8]) * voxel_volume).astype(int)
#    result = build_voxel_synapse_count(distmap, voxel_size, 1).raw
#    npt.assert_equal(result[0][:10],
#                     np.array([[0],
#                               [count_per_slice[0]],
#                               [count_per_slice[0]],
#                               [count_per_slice[1]],
#                               [count_per_slice[1]],
#                               [count_per_slice[2]],
#                               [count_per_slice[2]],
#                               [count_per_slice[2]],
#                               [count_per_slice[2]],
#                               [0]]))
