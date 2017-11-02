#!/usr/bin/env python
'''Test run of Projectionizer code for NSETM-206

The densities wanted span the whole y-extent, which is different than usual,
and are very low, which results in having to oversample quite a bit, or use
large voxel sizes
'''
import argparse
import logging
import math
import os

import numpy as np
import pandas as pd

from bluepy.api import Circuit
from voxcell import VoxelData

from SSCX_Thalamocortical_VPM_hex import (tiled_locations,
                                          assign_synapses, pick_synapses, prune,
                                          )
from projectionizer import nrnwriter, sscx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#From Srikanth
#    Please find attached the data sets for integration into the
#   projectionizer.
#
#   Methods.docx should give you an idea for what was done and the actual
#   data is in "Fiber length... .xls"
#
#   In Fiber length... .xls the data to be integrated is in CHAT
#   -> VARICOSITIES.

#All data are: Varicosities / 1000 um^3
#              mean          stdev
VARICOSITY = {'1': (0.4799547285, 0.0963170146),
              '2': (0.3869016127, 0.0679945021),
              '3': (0.3697457851, 0.0600613229),
              '4': (0.3740495687, 0.0383633721),
              '5': ((0.4851163218 + 0.3505663408) /2., 0.1089693436), # avg of L5a & L5b, var fm L5a
              #'L5a': (0.4851163218, 0.1089693436),
              #'L5b': (0.3505663408, 0.0663861683),
              '6': (0.3252380622, 0.0951497929)}


VOXEL_SIZE_UM = 50
OVERSAMPLE = 20.0


def build_voxel_synapse_count(voxel_size=VOXEL_SIZE_UM, oversample=OVERSAMPLE):
    raw = np.zeros(shape=(1, int(sscx.LAYER_BOUNDARIES[-1] // voxel_size), 1), dtype=np.int)

    for name in sscx.LAYERS:
        bottom = int(sscx.LAYER_STARTS[name] // voxel_size)
        top = int((sscx.LAYER_STARTS[name] + sscx.LAYER_THICKNESS[name]) // voxel_size)
        density = voxel_size ** 3 * oversample * VARICOSITY[name][0] / 1000
        print bottom, top, density
        raw[:, bottom:top, :] = math.ceil(density)

    return VoxelData(raw, [voxel_size] * 3, (0, 0, 0))


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', default='.',
                        help='Output directory')
    parser.add_argument('-p', '--parallelize', action='store_true',
                        help='Output directory')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='-v for INFO, -vv for DEBUG')

    return parser


def sample_synapses(map_):
    voxel_synapse_count = build_voxel_synapse_count()
    locations = tiled_locations(voxel_size=VOXEL_SIZE_UM)
    synapses = pick_synapses(locations, voxel_synapse_count, circuit, map_=map_)

    synapses.reset_index(inplace=True, drop=True)
    synapses.columns = map(str, synapses.columns)
    remove_cols = [u'Segment.X1', u'Segment.Y1', u'Segment.Z1',
                   u'Segment.X2', u'Segment.Y2', u'Segment.Z2']
    synapses.drop(remove_cols, axis=1, inplace=True)

    #set datatypes so less memory
    synapses['Section.ID'] = synapses['Section.ID'].values.astype(np.int32)
    synapses['Segment.ID'] = synapses['Segment.ID'].values.astype(np.int32)
    synapses['gid'] = synapses['gid'].values.astype(np.int32)
    synapses['segment_length'] = synapses['segment_length'].values.astype(np.float32)

    return synapses


def _get_ax():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def synapses_per_connection(synapses, output, name):
    fig, ax = _get_ax()
    bins = np.arange(50)
    g = synapses.groupby(['sgid', 'tgid'], sort=False).size()
    ax.hist(g, bins=bins)

    ax.set_title('Synapses per Connection: ' + name)

    path = os.path.join(output, name + '_synapses_per_connection.png')
    fig.savefig(path)


def plot_density(synapses, output, name, bin_size=50):
    def vol(df):
        xyz = list('xyz')
        vol_ = df[xyz].max().values - df[xyz].min().values
        vol_[1] = bin_size
        return np.prod(vol_)

    bins = np.arange(0, synapses.y.max(), bin_size)
    hist, edges = np.histogram(synapses.y.values, bins=bins)

    hist = hist / vol(synapses)

    fig, ax = _get_ax()
    ax.set_xlim(0, 2100)
    ax.set_xlabel('Layer depth um')
    ax.set_ylabel('Density (syn/um3)')
    ax.set_title('Density: ' + name)
    ax.bar(edges[:-1], hist, width=np.diff(edges), ec="k", align="edge")

    for k, v in sscx.LAYER_STARTS.items():
        ax.axvline(v)

    path = os.path.join(output, name + '_density.png')
    fig.savefig(path)


def write_feather_and_nrn(synapses, output, name, write_nrn=True):
    syn_path = os.path.join(output, name + '.feather')
    synapses.reset_index(inplace=True, drop=True)
    synapses.columns = map(str, synapses.columns)
    synapses.to_feather(syn_path)

    if write_nrn:
        nrn_path = os.path.join(output, name + '_nrn.h5')
        synapses['sgid'] += 1000000  # make sure we don't have a collision w/ a gid
        gb = synapses.groupby('tgid')
        nrnwriter.write_synapses(nrn_path, iter(gb), sscx.create_synapse_data)

    plot_density(synapses, output, name)
    synapses_per_connection(synapses, output, name)


def main(args):
    logging.basicConfig(level=(logging.WARNING,
                               logging.INFO,
                               logging.DEBUG)[min(args.verbose, 2)])

    BASE_CIRCUIT = ('/gpfs/bbp.cscs.ch/project/proj1/circuits/'
                    'SomatosensoryCxS1-v5.r0/O1/merged_circuit/')
    circuit = os.path.join(BASE_CIRCUIT, 'CircuitConfig')
    circuit = Circuit(circuit).v2
    #Note: original v5.r0 doesn't have an MVD3, so I converted my own
    circuit._config['cells'] = ('/gpfs/bbp.cscs.ch/project/proj30/mgevaert/'
                                'SomatosensoryCxS1-v5.r0_circuit.mvd3')

    map_ = map

    if args.parallelize:
        from map_parallelize import map_parallelize as map_
        from dask.distributed import Client
        client = Client()

    SAMPLE = False
    if SAMPLE:
        sample_synapses(map_)
        assigned_synapses = assign_synapses(synapses, map_=map_)

        p = 'segments_vox_%d_%dx_assigned' % (VOXEL_SIZE_UM, int(OVERSAMPLE))
        assigned_synapses.columns = map(str, assigned_synapses.columns)
        assigned_synapses.to_feather(os.path.join(args.output, p))
    else:
        p = '/gpfs/bbp.cscs.ch/home/gevaert/proj30/mgevaert/vari/segments_vox_50_20x_assigned.feather'
        assigned_synapses = pd.read_feather(p)

    assigned_synapses.rename(columns={'segment_length': 'location'}, inplace=True)

    write_feather_and_nrn(assigned_synapses, args.output,
                          '%dx_oversample_no_pruning' % int(OVERSAMPLE), write_nrn=False)
    write_feather_and_nrn(assigned_synapses.sample(frac=1. / OVERSAMPLE), args.output,
                          'no_oversample_no_pruning')

    pruned = assigned_synapses
    for name in ('prune_once', 'prune_twice', 'prune_thrice'):
        pruned = prune(pruned, circuit, True)
        del pruned['mtype']
        write_feather_and_nrn(pruned, args.output, name)


if __name__ == '__main__':
    PARSER = get_parser()
    main(PARSER.parse_args())
