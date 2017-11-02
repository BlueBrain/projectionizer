import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _make_hist(y, bins):
    hist, _ = np.histogram(y, bins=bins)
    return hist


def _get_ax():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def synapse_density(orig_data, keep_syn, distmap, bin_width=25):
    def vol(df):
        xz = list('xz')
        xz_extend = df[xz].max().values - df[xz].min().values
        return np.prod(xz_extend) * bin_width

    bins = np.arange(600, 1600, bin_width)

    df = pd.DataFrame(index=bins[:-1])
    df['Original'] = _make_hist(orig_data.y.values, bins) / vol(orig_data)
    df['New'] = _make_hist(keep_syn.y.values, bins) / vol(keep_syn)

    fig, ax = _get_ax()

    def draw_distmap(dm):
        values = np.array(dm)
        values = np.vstack((values[:, 0].repeat(2)[1:], values[:, 1].repeat(2)[:-1])).T
        ax.plot(values[:, 0], values[:, 1], 'r--', linewidth=2)

    draw_distmap(distmap[0])
    draw_distmap(distmap[1])

    ax.set_xlim(600, 1600)
    ax.set_ylim(0, 0.05)

    ax.set_xlabel('Layer depth um')
    ax.set_ylabel('Density (syn/um3)')

    ax2 = ax.twiny()
    df.plot(kind='bar', ax=ax2, sharey=True)

    # remove upper axis ticklabels
    ax2.set_xticklabels([])
    # set the limits of the upper axis to match the lower axis ones

    ax.set_title('Synapse density histogram')
    return fig


def syns_per_connection(orig_data, keep_syn):
    fig, ax = _get_ax()

    bins = np.arange(30)
    df = pd.DataFrame(index=bins[:-1])

    orig_counts = orig_data.groupby(['tgid', 'sgid']).size()
    df['Original'] = _make_hist(orig_counts, bins)

    new_counts = keep_syn.groupby(['tgid', 'sgid']).size()
    df['New'] = _make_hist(new_counts, bins)

    df.plot(kind='barh', ax=ax)
    ax.set_title('Synapse / connection')
    return fig


def plot_hexagon(ax, center=(0., 0.)):
    '''center in on `center`'''
    from mini_col_locations import hexagon
    points = hexagon()
    ax.plot(center[0] + points[:, 0], center[1] + points[:, 1], 'r-')


def plot_used_minicolumns(df, ax=None):
    ''''''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    from mini_col_locations import get_virtual_fiber_locations
    locations = get_virtual_fiber_locations()
    x, z = locations[:, 0], locations[:, 1]

    plot_hexagon(ax, center=(np.mean(x), np.mean(z)))
    ax.scatter(x, z, c='b')

    uniq = df.sgid.unique()
    ax.scatter(x[uniq], z[uniq], c='r')


def column_scatter_plots(df, prefix, fiber_locations=None):
    import mpl_scatter_density

    assert 'x' in df.columns
    x, y, z = df['x'].values, df['y'].values, df['z'].values

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    ax.scatter_density(x, z)
    plot_hexagon(ax, center=(np.mean(x), np.mean(z)))
    ax.set_title('Synapse locations: x/z')

    if fiber_locations is not None:
        ax.scatter(fiber_locations[:, 0], fiber_locations[:, 1], c='red', s=1)

    ax.set_xlabel('x location in um')
    ax.set_ylabel('z location in um')
    fig.savefig(prefix + '_synpases_xz.png')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    ax.scatter_density(x, y)
    ax.set_title('Synapse locations: x/y')
    ax.set_xlabel('x location in um')
    ax.set_ylabel('y location in um')
    fig.savefig(prefix + '_synpases_xy.png')
