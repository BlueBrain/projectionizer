'''helpers for creating plots'''
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from projectionizer import luigi_utils, sscx_hex

def _get_ax():
    '''return axis'''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def plot_synapses_per_connection(syns, output, name):
    '''plot_synapses_per_connection'''
    fig, ax = _get_ax()
    bins = np.arange(50)
    g = syns.groupby(['sgid', 'tgid'], sort=False).size()
    ax.hist(g, bins=bins)

    ax.set_title('syns per Connection: ' + name)

    path = os.path.join(output, name + '_synapses_per_connection.png')
    fig.savefig(path)


def plot_density(syns, output, name, layers, bin_size=50, save_csv=True):
    '''plot_density'''
    def vol(df):
        '''get volume'''
        xyz = list('xyz')
        vol_ = df[xyz].max().values - df[xyz].min().values
        vol_[1] = bin_size
        return np.prod(vol_)

    bins = np.arange(0, syns.y.max() + bin_size, bin_size)
    hist, edges = np.histogram(syns.y.values, bins=bins)

    hist = hist / vol(syns)

    fig, ax = _get_ax()
    ax.set_xlim(0, 2100)
    ax.set_xlabel('Layer depth um')
    ax.set_ylabel('Density (syn/um3)')
    ax.set_title('Density: ' + name)
    ax.bar(edges[:-1], hist, width=np.diff(edges), ec="k", align="edge")

    total = 0
    for layer_name, start in layers:
        total += start
        ax.axvline(total, label=str(layer_name), color='red')

    if save_csv:
        hist = pd.Series(hist, index=bins[:-1])
        path = os.path.join(output, name + '_density.csv')
        hist.to_csv(path, index_label='Depth (um)', header=['Density (syn/um^3)'])

    path = os.path.join(output, name + '_density.png')
    fig.savefig(path)


def plot_hex(syns, output, name, fibers_path, hex_apron_size, hex_edge_len):
    '''plot_hex'''
    fig, ax = _get_ax()
    ax.set_title('Hex Density: ' + name)
    ax.set_aspect('equal')

    ax.scatter(syns.x.values, syns.z.values, alpha=0.1, marker='.', s=1)

    all_fibers = sscx_hex.get_minicol_virtual_fibers(apron_size=hex_apron_size,
                                                     hex_edge_len=hex_edge_len,
                                                     locations_path=fibers_path)

    ax.scatter(all_fibers.x, all_fibers.z, color='red', marker='.', s=1)

    hex_points = sscx_hex.hexagon(hex_edge_len) + (syns.x.mean(), syns.z.mean())
    v = (0, 1, 2, 3, 4, 5, 6, 0)
    ax.plot(hex_points[v, 0], hex_points[v, 1], color='red')

    path = os.path.join(output, name + '_hex.png')
    fig.savefig(path)