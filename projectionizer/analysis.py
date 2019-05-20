'''Plotting module'''
import logging
import os
from itertools import chain, repeat
import yaml

import matplotlib
matplotlib.use('Agg')
# deal w/ using Agg backend
# pylint: disable=wrong-import-position

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluepy.v2.circuit import Circuit

from projectionizer.sscx_hex import get_minicol_virtual_fibers
from projectionizer.luigi_utils import CommonParams, RunAnywayTargetTempDir
from projectionizer.step_0_sample import FullSample, SynapseDensity
from projectionizer.step_2_prune import (ChooseConnectionsToKeep, CutoffMeans,
                                         ReducePrune)
from projectionizer.step_3_write import VirtualFibers, WriteAll
from projectionizer.utils import load, load_all, read_feather


L = logging.getLogger(__name__)
L.setLevel(logging.DEBUG)


def draw_layer_boundaries(ax, layers):
    '''draw layer boundaries as defined by `layers`'''
    total = 0
    for name, delta in layers:
        total += delta
        ax.axvline(x=total, color='green')
        ax.text(x=total, y=100, s='Layer %d' % name)
    ax.set_xlim([0, total])


def draw_distmap(ax, distmap, oversampling, linewidth=2):
    '''Draw expected density as a function of height'''
    def get_values(distmap):
        '''Stack distmap from both layers'''
        values = np.array(distmap)
        return np.vstack((values[:, 0].repeat(2)[1:], values[:, 1].repeat(2)[:-1])).T

    values = get_values(distmap[0])
    ax.plot(values[:, 0], values[:, 1] * oversampling, 'r--', linewidth=linewidth)
    values = get_values(distmap[1])
    ax.plot(values[:, 0], values[:, 1] * oversampling, 'r--',
            linewidth=linewidth, label='expected density')


def _make_hist(y, bins):
    '''Build a hist'''
    hist, _ = np.histogram(y, bins=bins)
    return hist


def _get_ax():
    '''Create new fig and returns fig, ax'''
    fig = plt.figure(figsize=(12.0, 10.0))
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def fill_voxels(voxel_like, coordinates):
    '''Fill voxel'''
    idx = pd.DataFrame(voxel_like.positions_to_indices(coordinates), columns=list('xyz'))
    counts_idx = idx.groupby(list('xyz')).size().reset_index()
    counts = np.zeros_like(voxel_like.raw, dtype=np.uint)
    counts[counts_idx['x'], counts_idx['y'], counts_idx['z']] = counts_idx.loc[:, 0]
    return voxel_like.with_data(counts)


def synapse_density_per_voxel(folder, synapses, layers, distmap, oversampling, prefix=''):
    '''2D-distribution: voxel height - voxel density'''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    title = "Synaptic density per voxel " + prefix
    ax.set_title(title)
    heights = load(os.path.join(folder, 'height.nrrd'))
    voxel_volume = np.prod(np.abs(heights.voxel_dimensions))
    counts = fill_voxels(heights, synapses[list('xyz')].values).raw / voxel_volume
    heights_counts = np.stack((heights.raw, counts), axis=3).reshape(-1, 2)
    heights_counts = heights_counts[heights_counts[:, 1] > 0]

    x = heights_counts[:, 0]
    y = heights_counts[:, 1]
    ax.hist2d(x, y, bins=40, label='synaptic density')

    draw_distmap(ax, distmap, oversampling, linewidth=1)
    draw_layer_boundaries(ax, layers)

    ax.set_xlabel('Voxel height (um)')
    ax.set_ylabel(r'Synaptical density ($\mathregular{um^{-3}}$)')
    fig.savefig(os.path.join(folder, '{}_density_per_voxel.png'.format(prefix)))


def remove_synapses_with_sgid(synapses, sgids):
    '''Used to remove synapses from apron region'''
    remove_idx = np.in1d(synapses.sgid.values, sgids)
    synapses = synapses[~remove_idx]
    return synapses


def synapse_density(keep_syn, distmap, layers, bin_width=25, oversampling=1, folder='.'):
    '''Plot synaptic density profile'''
    def vol(df):
        '''Get volume'''
        xz = list('xz')
        xz_extend = df[xz].max().values - df[xz].min().values
        return np.prod(xz_extend) * bin_width

    fig, ax = _get_ax()

    draw_distmap(ax, np.array(distmap), oversampling)
    draw_layer_boundaries(ax, layers)

    bins = np.arange(keep_syn.y.min(), keep_syn.y.max(), bin_width)
    height = _make_hist(keep_syn.y.values, bins) / vol(keep_syn)

    ax.bar(x=bins[:-1], height=height, width=bin_width, align='edge')

    ax.set_xlabel('Layer depth um')
    ax.set_ylabel('Density (syn/um3)')

    ax.set_title('Synapse density histogram')
    fig.savefig(os.path.join(folder, 'density.png'))

    return fig


def fraction_pruned_vs_height(folder, n_chunks):
    '''Plot how many synapses are pruned vs height'''
    kept = read_feather('{}/choose-connections-to-keep.feather'.format(folder))
    chunks = list()
    for i in range(n_chunks):
        df = read_feather('{}/sample-chunk-{}.feather'.format(folder, i))
        sgid = read_feather('{}/fiber-assignment-{}.feather'.format(folder, i))
        chunks.append(df[['tgid', 'y']].join(sgid))

    fat = pd.merge(pd.concat(chunks),
                   kept[['sgid', 'tgid', 'kept']],
                   left_on=['sgid', 'tgid'], right_on=['sgid', 'tgid'])
    step = 100
    bins = np.linspace(fat.y.min(), fat.y.max() + step, step)
    bin_center = 0.5 * (bins[1:] + bins[:-1])

    s = pd.cut(fat.y, bins)
    g = fat.groupby(s.cat.rename_categories(bin_center))
    g['kept'].mean().plot(kind='bar')


def syns_per_connection_per_mtype(choose_connections, cutoffs, folder):
    '''2D-Plot of number of synapse per connection distribution VS mtype'''
    fig, ax = _get_ax()
    ax.set_title('Number of synapses/connection for each mtype')
    bins_syn = np.arange(50)
    grp = choose_connections.groupby('mtype')
    mtypes = sorted(grp.groups)
    mtype_connection_count = np.array(list(chain.from_iterable(
        zip(repeat(i), grp['0'].get_group(mtype)) for i, mtype in enumerate(mtypes))))
    x = mtype_connection_count[:, 0]
    y = mtype_connection_count[:, 1]
    bins = (np.arange(len(mtypes)), bins_syn)
    ax.hist2d(x, y, bins=bins, norm=colors.LogNorm())
    mean_connection_count_per_mtype = grp.mean().loc[:, '0']
    plt.step(bins[0], mean_connection_count_per_mtype,
             where='post', color='red', label='Mean value')
    plt.step(bins[0], cutoffs.sort_values('mtype').cutoff,
             where='post', color='black', label='Cutoff')
    plt.xticks(bins[0] + 0.5, mtypes, rotation=90)
    plt.legend()
    fig.savefig(os.path.join(folder, 'syns_per_connection_per_mtype.png'))


def syns_per_connection(choose_connections, cutoffs, folder):
    '''Plot the number of synapses per connection'''
    fig, ax = _get_ax()
    max_synapses = 50
    bins_syn = np.linspace(0, max_synapses, 50)
    choose_connections[choose_connections.kept].loc[:, '0'].hist(bins=bins_syn)

    ax.set_title('Synapse / connection')
    fig.savefig(os.path.join(folder, 'syns_per_connection.png'))
    syns_per_connection_per_mtype(choose_connections, cutoffs, folder)

    fig, ax = _get_ax()
    l4_pc_cells = choose_connections[choose_connections.mtype.isin(
        ['L4_PC', 'L4_UPC', 'L4_TPC'])]
    l4_pc_cells[choose_connections.kept].loc[:, '0'].hist(bins=np.arange(max_synapses))
    mean_value = l4_pc_cells[choose_connections.kept].loc[:, '0'].mean()
    plt.axvline(x=mean_value, color='red')
    ax.set_xlabel('Synapse count per connection')
    ax.set_title('Number of synapses/connection for L4_PC cells\nmean value = {}'
                 .format(mean_value))
    fig.savefig(os.path.join(folder, 'syns_per_connection_L4_PC.png'))

    fig, ax = _get_ax()
    l4_pc_cells.loc[:, '0'].hist(bins=np.arange(50))
    mean_value = l4_pc_cells.loc[:, '0'].mean()
    plt.axvline(x=mean_value, color='red')
    ax.set_xlabel('Synapse count per connection')
    ax.set_title(
        'Number of synapses/connection for L4_PC cells pre-pruning\nmean value = {}'
        .format(mean_value))
    fig.savefig(os.path.join(folder, 'syns_per_connection_L4_PC_pre_pruning.png'))


def efferent_neuron_per_fiber(df, fibers, folder):
    '''1D distribution of the number of neuron connected to a fiber averaged on all fibers'''
    title = "Efferent neuron count (averaged)"
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    neuron_efferent_count = df.groupby('sgid').tgid.nunique()
    ax.set_xlabel('Efferent neuron count for each fiber')
    plt.hist(neuron_efferent_count, bins=100, label='This work')

    plt.legend()
    fig.savefig(os.path.join(folder, 'efferent_count_1d.png'))

    title = "Efferent neuron count map"
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    df = fibers.join(neuron_efferent_count).fillna(0)
    plt.scatter(df.x, df.z, s=80, c=df.tgid)

    ax.set_xlabel('X fiber coordinate')
    ax.set_ylabel('Z fiber coordinate')

    fig.savefig(os.path.join(folder, 'efferent_count_2d.png'))


def innervation_width(pruned, circuit_config, folder):
    '''Innervation width

    mean distance between synapse and connected fiber
    '''
    c = Circuit(circuit_config)
    fig, ax = _get_ax()
    pruned.groupby('sgid').tgid.apply(lambda tgid: c.cells.get(
        tgid).x).groupby('sgid').std().hist(bins=40)
    ax.set_title('Innervation width (along X axis)')
    fig.savefig('{}/innervation_x_width.png'.format(folder))


class Analyse(CommonParams):
    '''Run the analysis'''

    def requires(self):  # pragma: no cover
        if self.geometry != 's1':
            tasks = (ReducePrune,
                     FullSample,
                     ChooseConnectionsToKeep,
                     CutoffMeans,
                     SynapseDensity,
                     VirtualFibers,
                     )
        else:
            tasks = (ReducePrune,
                     ChooseConnectionsToKeep,
                     CutoffMeans,
                     SynapseDensity,
                     VirtualFibers,
                     )
        return [self.clone(task) for task in tasks]

    def run(self):  # pragma: no cover
        if self.geometry != 's1':
            pruned, sampled, connections, cutoffs, distmap, fibers = load_all(self.input())
            locations_path = self.load_data(self.hex_fiber_locations)
            all_fibers = get_minicol_virtual_fibers(apron_size=self.hex_apron_size,
                                                    hex_edge_len=self.hex_side,
                                                    locations_path=locations_path)
            pruned_no_edge = remove_synapses_with_sgid(pruned,
                                                       all_fibers[all_fibers['apron']].index)
            fraction_pruned_vs_height(self.folder, self.n_total_chunks)
            innervation_width(pruned, self.circuit_config, self.folder)
            layers = yaml.load(self.layers)
            synapse_density_per_voxel(self.folder,
                                      sampled,
                                      layers,
                                      distmap,
                                      self.oversampling,
                                      'sampled')
        else:
            pruned, connections, cutoffs, distmap, fibers = load_all(self.input())
            pruned_no_edge = pruned
            layers = []

        connections.sgid += self.sgid_offset

        synapse_density_per_voxel(self.folder, pruned_no_edge, layers, distmap, 1., 'pruned')
        synapse_density(pruned_no_edge, distmap, layers, folder=self.folder)
        syns_per_connection(connections, cutoffs, self.folder)

        efferent_neuron_per_fiber(pruned, fibers, self.folder)

        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)


class DoAll(CommonParams):
    """Launch the full projectionizer pipeline"""

    def requires(self):
        return self.clone(ReducePrune), self.clone(Analyse), self.clone(WriteAll)

    def run(self):  # pragma: no cover
        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)
