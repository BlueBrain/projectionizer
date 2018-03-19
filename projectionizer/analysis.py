'''Plotting module'''
import logging
import os
from itertools import chain, repeat

import matplotlib
matplotlib.use('Agg')
# deal w/ using Agg backend
# pylint: disable=wrong-import-position

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluepy.v2.circuit import Circuit
from luigi import Parameter

from projectionizer.sscx_hex import get_minicol_virtual_fibers
from projectionizer.luigi_utils import CommonParams, RunAnywayTargetTempDir
from projectionizer.step_0_sample import FullSample, SynapseDensity
from projectionizer.step_2_prune import (ChooseConnectionsToKeep, CutoffMeans,
                                         ReducePrune)
from projectionizer.step_3_write import VirtualFibers, WriteAll
from projectionizer.utils import load, load_all


L = logging.getLogger(__name__)
L.setLevel(logging.DEBUG)


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


def synapse_density_per_voxel(folder, synapses, distmap, oversampling, prefix=''):
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
    plt.hist2d(x, y, bins=40, label='synaptic density')
    draw_distmap(ax, distmap, oversampling, linewidth=1)
    ax.set_xlabel('Voxel height (um)')
    ax.set_ylabel(r'Synaptical density ($\mathregular{um^{-3}}$)')
    cbar = plt.colorbar()
    cbar.set_label('# of voxels')
    plt.legend()
    fig.savefig(os.path.join(folder, '{}_density_per_voxel.png'.format(prefix)))


def remove_synapses_with_sgid(synapses, sgids):
    '''Used to remove synapses from apron region'''
    remove_idx = np.in1d(synapses.sgid.values, sgids)
    synapses = synapses[~remove_idx]
    return synapses


def synapse_density(orig_data, keep_syn, distmap, bin_width=25, oversampling=1, folder='.'):
    '''Plot synaptic density profile'''
    def vol(df):
        '''Get volume'''
        xz = list('xz')
        xz_extend = df[xz].max().values - df[xz].min().values
        return np.prod(xz_extend) * bin_width

    bins = np.arange(keep_syn.y.min(), keep_syn.y.max(), bin_width)

    df = pd.DataFrame(index=bins[:-1])
    if orig_data is not None:
        df['Original'] = _make_hist(orig_data.y.values, bins) / vol(orig_data)
    df['New'] = _make_hist(keep_syn.y.values, bins) / vol(keep_syn)

    fig, ax = _get_ax()

    draw_distmap(ax, np.array(distmap), oversampling)

    ax.set_xlabel('Layer depth um')
    ax.set_ylabel('Density (syn/um3)')

    ax2 = ax.twiny()
    df.plot(kind='bar', ax=ax2, sharey=True)

    # remove upper axis ticklabels
    ax2.set_xticklabels([])

    ax.set_title('Synapse density histogram')
    fig.savefig(os.path.join(folder, 'density.png'))

    return fig


def fraction_pruned_vs_height(folder, n_chunks):
    '''Plot how many synapses are pruned vs height'''
    kept = pd.read_feather('{}/choose-connections-to-keep.feather'.format(folder))
    chunks = list()
    for i in range(n_chunks):
        df = pd.read_feather('{}/sample-chunk-{}.feather'.format(folder, i))
        sgid = pd.read_feather('{}/fiber-assignment-{}.feather'.format(folder, i))
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
    plt.hist2d(x, y, bins=bins, norm=colors.LogNorm())
    mean_connection_count_per_mtype = grp.mean().loc[:, '0']
    plt.step(bins[0], mean_connection_count_per_mtype,
             where='post', color='red', label='Mean value')
    plt.step(bins[0], cutoffs.sort_values('mtype').cutoff,
             where='post', color='black', label='Cutoff')
    plt.xticks(bins[0] + 0.5, mtypes, rotation=90)
    plt.colorbar()
    plt.legend()
    fig.savefig(os.path.join(folder, 'syns_per_connection_per_mtype.png'))


def syns_per_connection(orig_data, choose_connections, cutoffs, folder):
    '''Plot the number of synapses per connection'''
    fig, ax = _get_ax()
    max_synapses = 50
    bins_syn = np.linspace(0, max_synapses, 50)
    choose_connections[choose_connections.kept].loc[:, '0'].hist(bins=bins_syn)

    if orig_data is not None:
        orig_counts = orig_data.groupby(['tgid', 'sgid']).size()
        _make_hist(orig_counts, bins_syn)

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


def efferent_neuron_per_fiber(df, fibers, folder, cell_data=True):
    '''1D distribution of the number of neuron connected to a fiber averaged on all fibers'''
    title = "Efferent neuron count (averaged)"
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    neuron_efferent_count = df.groupby('sgid').tgid.nunique()
    ax.set_xlabel('Efferent neuron count for each fiber')
    plt.hist(neuron_efferent_count, bins=100, label='This work')

    if cell_data:
        data_cell_paper = [
            [752.7687, 0.9569],
            [767.9064, 5.9849],
            [786.2024, 1.0307],
            [803.2772, 1.9899],
            [817.4414, 1.9889],
            [832.0066, 6.9411],
            [849.5682, 6.9146],
            [865.6816, 2.9968],
            [882.3453, 5.8766],
            [897.9659, 9.8683],
            [914.7236, 13.9106],
            [930.5917, 6.9602],
            [948.5542, 11.8868],
            [963.4527, 6.9581],
            [978.1037, 5.9716],
            [995.5856, 4.9597],
            [1011.2082, 1.9767],
            [1027.4812, 0.0299],
            [1042.8563, 0.9891],
            [1059.2888, 1.0134],
            [1073.9399, 0.0269],
            [1084.7088, 0.0768],
        ]
        x_cell_paper, y_cell_paper = zip(*data_cell_paper)
        plt.plot(x_cell_paper, np.array(y_cell_paper) * 2.4, label='Cell paper data (norm.)')
    plt.legend()
    fig.savefig(os.path.join(folder, 'efferent_count_1d.png'))

    title = "Efferent neuron count map"
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    df = fibers.join(neuron_efferent_count).fillna(0)
    plt.scatter(df.x, df.z, s=80, c=df.tgid)
    cbar = plt.colorbar()
    ax.set_xlabel('X fiber coordinate')
    ax.set_ylabel('Z fiber coordinate')
    cbar.set_label('# of efferent neurons')
    fig.savefig(os.path.join(folder, 'efferent_count_2d.png'))


def innervation_width(pruned, circuit_config, folder):
    '''Innervation width

    mean distance between synapse and connected fiber
    '''
    # '/gpfs/bbp.cscs.ch/project/proj64/circuits/O1.v5/20171107/CircuitConfig'
    c = Circuit(circuit_config)
    fig, ax = _get_ax()
    pruned.groupby('sgid').tgid.apply(lambda tgid: c.cells.get(
        tgid).x).groupby('sgid').std().hist(bins=40)
    ax.set_title('Innervation width (along X axis)')
    fig.savefig('{}/innervation_x_width.png'.format(folder))


class Analyse(CommonParams):
    '''Run the analysis'''
    original_data = Parameter(
        default=('/gpfs/bbp.cscs.ch/project/proj30/mgevaert/'
                 'ncsThalamocortical_VPM_tcS2F_2p6_ps.feather'))

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
            original = load(self.original_data)

            fraction_pruned_vs_height(self.folder, self.n_total_chunks)
            innervation_width(pruned, self.circuit_config, self.folder)
            synapse_density_per_voxel(self.folder, sampled, distmap, self.oversampling, 'sampled')
        else:
            pruned, connections, cutoffs, distmap, fibers = load_all(self.input())
            pruned_no_edge = pruned
            original = None

        connections.sgid += self.sgid_offset

        synapse_density_per_voxel(self.folder, pruned_no_edge, distmap, 1., 'pruned')
        synapse_density(original, pruned_no_edge, distmap, folder=self.folder)
        syns_per_connection(original, connections, cutoffs, self.folder)

        efferent_neuron_per_fiber(pruned, fibers, self.folder, cell_data=self.geometry != 's1')

        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)


class DoAll(CommonParams):
    """Launch the full projectionizer pipeline"""

    def requires(self):
        return self.clone(WriteAll), self.clone(Analyse)

    def run(self):  # pragma: no cover
        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)
