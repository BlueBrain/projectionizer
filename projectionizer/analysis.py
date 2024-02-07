"""Plotting module"""

import json
import logging
import re
from itertools import chain, repeat

import matplotlib

matplotlib.use("Agg")  # pylint: disable=wrong-import-position

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluepy import Cell, Circuit
from matplotlib import colors

from projectionizer.fiber_simulation import get_region_ids
from projectionizer.luigi_utils import CommonParams, JsonTask, RunAnywayTargetTempDir
from projectionizer.step_0_sample import FullSample, Height, SynapseDensity
from projectionizer.step_1_assign import VirtualFibers
from projectionizer.step_2_prune import (
    ChooseConnectionsToKeep,
    CutoffMeans,
    ReducePrune,
)
from projectionizer.utils import (
    convert_layer_to_PH_format,
    load,
    load_all,
    read_feather,
)

L = logging.getLogger(__name__)
L.setLevel(logging.DEBUG)

SYNS_CONN_MAX_BINS = 50
FIBER_THRESHOLD = 0.8
DENSITY_THRESHOLD = 0.1


def draw_layer_boundaries(ax, layer_thickness):
    """draw layer boundaries as defined by `layer_thickness`"""
    total = 0
    for name, delta in layer_thickness:
        total += delta
        ax.axvline(x=total, color="green")
        ax.text(x=total, y=100, s=f"Layer {name}")
    ax.set_xlim([0, total])


def draw_distmap(ax, distmap, oversampling, linewidth=2):
    """Draw expected density as a function of height"""

    def get_values(distmap):
        """Stack `distmap` from both layers"""
        values = np.array(distmap)
        return np.vstack((values[:, 0].repeat(2)[1:], values[:, 1].repeat(2)[:-1])).T

    values = get_values(distmap[0])
    ax.plot(values[:, 0], values[:, 1] * oversampling, "r--", linewidth=linewidth)

    # if we have a reference/expected dataset to compare to, display that as well
    if len(distmap) == 2:
        values = get_values(distmap[1])
        ax.plot(
            values[:, 0],
            values[:, 1] * oversampling,
            "r--",
            linewidth=linewidth,
            label="expected density",
        )


def _make_hist(y, bins):
    """Build a hist"""
    hist, _ = np.histogram(y, bins=bins)
    return hist


def _get_ax():
    """Create new fig and returns fig, ax"""
    fig = plt.figure(figsize=(12.0, 10.0))
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def fill_voxels(voxel_like, coordinates):
    """Fill voxel"""
    idx = pd.DataFrame(voxel_like.positions_to_indices(coordinates), columns=list("xyz"))
    counts_idx = idx.groupby(list("xyz")).size().reset_index()
    counts = np.zeros_like(voxel_like.raw, dtype=np.uint)
    counts[counts_idx["x"], counts_idx["y"], counts_idx["z"]] = counts_idx.loc[:, 0]
    return voxel_like.with_data(counts)


def synapse_density_per_voxel(folder, synapses, layer_thickness, distmap, oversampling, prefix=""):
    # pylint: disable=too-many-locals
    """2D-distribution: voxel height - voxel density"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    title = "Synaptic density per voxel " + prefix
    ax.set_title(title)
    heights = load(folder / "height.nrrd")
    voxel_volume = np.prod(np.abs(heights.voxel_dimensions))
    counts = fill_voxels(heights, synapses[list("xyz")].values).raw / voxel_volume
    heights_counts = np.stack((heights.raw, counts), axis=3).reshape(-1, 2)
    heights_counts = heights_counts[heights_counts[:, 1] > 0]
    np.nan_to_num(heights_counts, 0)
    total_height = np.array(layer_thickness)[:, 1].astype(float)
    total_height = total_height.sum()

    # Get the counts and heights for the distribution, fill the rest with zeros
    x = relative_height_to_absolute(heights_counts[:, 0], layer_thickness)
    y = heights_counts[:, 1]
    x_min_fill = np.array(range(int(np.min(heights_counts))))
    x_max_fill = np.array(range(int(np.max(heights_counts)), int(total_height)))
    y_min_fill = np.zeros_like(x_min_fill)
    y_max_fill = np.zeros_like(x_max_fill)
    x = np.hstack((x_min_fill, x, x_max_fill))
    y = np.hstack((y_min_fill, y, y_max_fill))

    ax.hist2d(x, y, bins=40, label="synaptic density")

    distmap = distmap_with_heights(distmap, layer_thickness)
    draw_distmap(ax, distmap, oversampling, linewidth=1)
    draw_layer_boundaries(ax, layer_thickness)

    ax.set_xlabel("Voxel height (um)")
    ax.set_ylabel(r"Synaptical density ($\mathregular{um^{-3}}$)")
    fig.savefig(folder / f"{prefix}_density_per_voxel.png")


def remove_synapses_with_sgid(synapses, sgids):
    """Used to remove synapses from apron region"""
    remove_idx = np.in1d(synapses.sgid.values, sgids)
    synapses = synapses[~remove_idx]
    return synapses


def relative_height_to_absolute(height, layer_thickness):
    """Converts relative height in layer to absolute height"""
    layer_thickness = np.array(layer_thickness)
    ind = np.floor(height)
    # Change nan indices to 0, they will be converted back to nan
    ind = np.nan_to_num(ind, 0).astype(int)
    fractions = height % 1
    # just a hack when last layer with fraction of 1.0 ends in L = L_max+1
    # (e.g., Layer 6 + 1.0 fraction ends up as 7 in heights)
    layer_heights = np.hstack([layer_thickness[:, 1], 0]).astype(float)
    cum_layer_height = np.hstack([0, np.cumsum(layer_heights)])

    return cum_layer_height[ind] + layer_heights[ind] * fractions


def distmap_with_heights(distmap, layer_thickness):
    """Append and return the `distmap` with absolute heights"""
    heights = []
    for dist in np.array(distmap):
        dist = np.array(dist)
        absolute_heights = relative_height_to_absolute(dist[:, 0], layer_thickness)
        heights.append(np.transpose([absolute_heights, dist[:, 1]]))

    return heights


def synapse_heights(full_sample, atlas, folder="."):
    """Plots a histogram of the heights"""
    distance = atlas.load_data("[PH]y")
    xyz = full_sample.sample(frac=0.01)[list("xyz")].to_numpy()
    idx = distance.positions_to_indices(xyz)
    dist = distance.raw[tuple(idx.T)]

    fig, ax = _get_ax()
    bin_width = dist.max() / 99
    bins = np.arange(0, dist.max(), bin_width)
    height = _make_hist(dist, bins)
    ax.bar(x=bins[:-1], height=height, width=bin_width, align="edge")
    ax.set_title("Synapse height histogram")
    ax.set_xlabel("Height")
    ax.set_ylabel("Number of synapses")
    fig.savefig(folder / "synapse_heights.png")


def synapse_density(keep_syn, distmap, layer_thickness, bin_width=25, oversampling=1, folder="."):
    """Plot synaptic density profile"""

    def vol(df):
        """Get volume"""
        xz = list("xz")
        xz_extend = df[xz].max().values - df[xz].min().values
        return np.prod(xz_extend) * bin_width

    fig, ax = _get_ax()

    dmap = distmap_with_heights(distmap, layer_thickness)
    draw_distmap(ax, np.array(dmap), oversampling)
    draw_layer_boundaries(ax, layer_thickness)

    bins = np.arange(keep_syn.y.min(), keep_syn.y.max(), bin_width)
    height = _make_hist(keep_syn.y.values, bins) / vol(keep_syn)

    ax.bar(x=bins[:-1], height=height, width=bin_width, align="edge")

    ax.set_xlabel("Layer depth um")
    ax.set_ylabel("Density (syn/um3)")

    ax.set_title("Synapse density histogram")
    fig.savefig(folder / "density.png")

    return fig


def fraction_pruned_vs_height(folder, n_chunks):
    """Plot how many synapses are pruned vs height"""
    kept = read_feather(f"{folder}/choose-connections-to-keep.feather")
    chunks = []
    for i in range(n_chunks):
        df = read_feather(f"{folder}/sample-chunk-{i}.feather")
        sgid = read_feather(f"{folder}/fiber-assignment-{i}.feather")
        chunks.append(df[["tgid", "y"]].join(sgid))

    fat = pd.merge(
        pd.concat(chunks),
        kept[["sgid", "tgid", "kept"]],
        left_on=["sgid", "tgid"],
        right_on=["sgid", "tgid"],
    )
    step = 100
    bins = np.linspace(fat.y.min(), fat.y.max() + step, step)
    bin_center = 0.5 * (bins[1:] + bins[:-1])

    s = pd.cut(fat.y, bins)
    g = fat.groupby(s.cat.rename_categories(bin_center))
    fig = g["kept"].mean().plot(kind="bar").get_figure()
    fig.savefig(folder / "fraction_pruned_vs_height.png")


def syns_per_connection_per_mtype(choose_connections, cutoffs, folder):
    """2D-Plot of number of synapse per connection distribution VS mtype"""
    fig, ax = _get_ax()
    ax.set_title("Number of synapses/connection for each mtype")
    bins_syn = np.arange(SYNS_CONN_MAX_BINS)
    grp = choose_connections.groupby("mtype")
    mtypes = sorted(grp.groups)
    mtype_connection_count = np.array(
        list(
            chain.from_iterable(
                zip(repeat(i), grp["connection_size"].get_group(mtype))
                for i, mtype in enumerate(mtypes)
            )
        )
    )
    x = mtype_connection_count[:, 0]
    y = mtype_connection_count[:, 1]
    bins = (np.arange(len(mtypes)), bins_syn)
    ax.hist2d(x, y, bins=bins, norm=colors.LogNorm())
    mean_connection_count_per_mtype = grp.mean().loc[:, "connection_size"]
    plt.step(
        bins[0], mean_connection_count_per_mtype, where="post", color="red", label="Mean value"
    )
    plt.step(
        bins[0], cutoffs.sort_values("mtype").cutoff, where="post", color="black", label="Cutoff"
    )
    plt.xticks(bins[0] + 0.5, mtypes, rotation=90)
    plt.legend()
    fig.savefig(folder / "syns_per_connection_per_mtype.png")


def syns_per_connection(choose_connections, cutoffs, folder, target_mtypes):
    """Plot the number of synapses per connection"""

    syns_per_connection_per_mtype(choose_connections, cutoffs, folder)

    bins_syn = np.arange(SYNS_CONN_MAX_BINS)

    fig, ax = _get_ax()
    choose_connections[choose_connections.kept].loc[:, "connection_size"].hist(bins=bins_syn)
    mean_value = choose_connections[choose_connections.kept].loc[:, "connection_size"].mean()
    plt.axvline(x=mean_value, color="red")
    ax.set_title(f"Synapse / connection: {mean_value}")
    fig.savefig(folder / "syns_per_connection.png")

    fig, ax = _get_ax()
    target_cells = choose_connections[choose_connections.mtype.isin(target_mtypes)]
    target_cells[choose_connections.kept].loc[:, "connection_size"].hist(bins=bins_syn)
    mean_value = target_cells[choose_connections.kept].loc[:, "connection_size"].mean()
    plt.axvline(x=mean_value, color="red")
    ax.set_xlabel("Synapse count per connection")
    ax.set_title(
        f"Number of synapses/connection for {target_mtypes} cells\nmean value = {mean_value}"
    )
    fig.savefig(folder / "syns_per_connection_checked.png")

    fig, ax = _get_ax()
    target_cells.loc[:, "connection_size"].hist(bins=bins_syn)
    mean_value = target_cells.loc[:, "connection_size"].mean()
    plt.axvline(x=mean_value, color="red")
    ax.set_xlabel("Synapse count per connection")
    ax.set_title(
        f"Number of synapses/connection for {target_mtypes} cells pre-pruning\n"
        f"mean value = {mean_value}"
    )
    fig.savefig(folder / "syns_per_connection_checked_pre_pruning.png")


def efferent_neuron_per_fiber(df, fibers, folder):
    """1D distribution of the number of neuron connected to a fiber averaged on all fibers"""
    title = "Efferent neuron count (averaged)"
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    neuron_efferent_count = df.groupby("sgid").tgid.nunique()
    ax.set_xlabel("Efferent neuron count for each fiber")
    plt.hist(neuron_efferent_count, bins=100, label="This work")

    plt.legend()
    fig.savefig(folder / "efferent_count_1d.png")

    title = "Efferent neuron count map"
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    df = fibers.join(neuron_efferent_count)
    plt.scatter(df.x, df.z, s=20, c=df.tgid)

    ax.set_xlabel("X fiber coordinate")
    ax.set_ylabel("Z fiber coordinate")
    ax.axis("equal")

    fig.savefig(folder / "efferent_count_2d.png")


def innervation_width(pruned, circuit_config, folder):
    """Innervation width

    mean distance between synapse and connected fiber
    """
    c = Circuit(circuit_config)
    fig, ax = _get_ax()
    pruned.groupby("sgid").tgid.apply(lambda tgid: c.cells.get(tgid).x).groupby("sgid").std().hist(
        bins=40
    )
    ax.set_title("Innervation width (along X axis)")
    fig.savefig(f"{folder}/innervation_x_width.png")


def thalamo_cortical_cells_per_fiber(pruned, folder):
    """Plots the histogram of thalamo-cortical cells per fiber"""
    # Thalamo-cortical sources
    n_fibers = pruned.sgid.nunique()
    counts = pruned.groupby("sgid")["tgid"].nunique().to_numpy()

    # Plot number of efferent thalamo-cortical target cells per fiber (Fig. 13C)
    bar_pos = 0.75
    plt.figure(figsize=(8, 6))
    plt.gcf().patch.set_facecolor("w")
    plt.hist(counts, range=(min(counts), max(counts)), bins=50, rwidth=0.75, color="k")
    plt.ylim([0, np.ceil(max(plt.ylim()) / 100) * 100])
    plt.plot(np.mean(counts), bar_pos * max(plt.ylim()), "o", color="gray")
    plt.plot(
        [np.mean(counts) - np.std(counts), np.mean(counts) + np.std(counts)],
        np.full(2, bar_pos * max(plt.ylim())),
        "|-",
        color="gray",
        linewidth=1.0,
    )
    plt.text(
        np.mean(counts),
        (bar_pos + 0.025) * max(plt.ylim()),
        f"{np.mean(counts):.2f}",
        color="gray",
        va="bottom",
        ha="center",
    )
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.title(f"Thalamo-cortical projections (N={n_fibers} fibers)")
    plt.xlabel("#Efferent neurons per thalamic fiber")
    plt.ylabel("Count")
    plt.savefig(folder / "tc_efferents.png", dpi=150)


def distribution_synapses_per_connection(pruned, folder):
    """Plots the histogram of synapses per connection"""

    # Compute number of synapses per connection
    counts = pruned.groupby(["sgid", "tgid"])["tgid"].count().values

    # Plot distribution of overall number of synapses per connection
    bar_pos = 0.75
    bar_colors = ["b"]
    mean_colors = ["cornflowerblue"]
    plt.figure(figsize=(8, 6))
    plt.gcf().patch.set_facecolor("w")
    plt.hist(counts, range=(0, 50), bins=50, rwidth=0.75, color=bar_colors[0])
    plt.ylim([0, np.ceil(max(plt.ylim()) / 1e6) * 1e6])
    plt.plot(np.mean(counts), bar_pos * max(plt.ylim()), "o", color=mean_colors[0])
    plt.plot(
        [np.mean(counts) - np.std(counts), np.mean(counts) + np.std(counts)],
        np.full(2, bar_pos * max(plt.ylim())),
        "|-",
        color=mean_colors[0],
        linewidth=1.0,
    )
    plt.text(
        np.mean(counts),
        (bar_pos + 0.025) * max(plt.ylim()),
        f"{np.mean(counts):.2f}",
        color=mean_colors[0],
        va="bottom",
        ha="center",
    )
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.title("Thalamo-cortical projections")
    plt.xlabel("#Synapses per connection")
    plt.ylabel("Count")
    plt.savefig(folder / "synapses_per_connection_overall.png", dpi=150)


def _synapses_per_connection_stats(pruned, cells, reg, sclass, lay):
    """Get synapses per connection for region, synapse class and layer.
    Return the connections, mean and standard deviation."""
    # NOTE: A lot of corner cases. E.g., for some columns, the format of the region is
    # <region>;<layer>, while for some it's plain <region>. This could be fixed with e.g.,
    # region_subregion_format introduced in WM by Mike.
    if re.match(r"L\d", lay):
        lay = int(lay[-1])
    elif lay.isdigit():
        lay = int(lay)
    if re.match(r"mc.*_Column", reg):
        # if region is `mc<N>_Column`, it can be `mc<N>;<layer>` or `mc<N>_Column`
        reg = [reg, re.sub("(mc.*)_Column", r"\g<1>", reg) + f";{lay}"]
    else:
        reg = [reg]
    tids_tmp = cells[
        np.logical_and(
            cells.region.isin(reg),
            np.logical_and(cells.synapse_class == sclass, cells.layer == lay),
        )
    ].index
    syn_prop_sel = pruned[np.in1d(pruned.tgid, tids_tmp)]
    if syn_prop_sel.size > 0:
        conns_tmp = syn_prop_sel.groupby(["tgid", "sgid"]).size().values
        return conns_tmp, conns_tmp.mean(), conns_tmp.std()

    return [], np.nan, np.nan


def distribution_synapses_per_connection_per_layer(pruned, regions, layers, circuit_config, folder):
    """Plots histograms of synapses per connection for each layer, region and synapse class."""
    # pylint: disable=too-many-locals
    c = Circuit(circuit_config)
    cells = c.cells.get(
        properties={Cell.REGION, Cell.X, Cell.Y, Cell.Z, Cell.LAYER, Cell.SYNAPSE_CLASS}
    )
    syn_classes = cells.synapse_class.unique().tolist()
    syn_classes.sort()

    # Plot separate results
    bar_pos = [0.9, 0.7]
    bar_colors = ["b", "g"]
    mean_colors = ["cornflowerblue", "mediumseagreen"]
    for sidx, sclass in enumerate(syn_classes):
        plt.figure(figsize=(12, 8))
        plt.gcf().patch.set_facecolor("w")
        for lidx, layer in enumerate(layers):
            for ridx, region in enumerate(regions):
                # Extract number of synapses per connection for each region/layer/synapse class
                conns, conns_mn, conns_sd = _synapses_per_connection_stats(
                    pruned, cells, region, sclass, layer
                )
                plt.subplot(len(layers), len(regions), lidx * len(regions) + ridx + 1)
                plt.hist(
                    conns,
                    range=(0, 50),
                    bins=25,
                    rwidth=0.75,
                    color=bar_colors[sidx],
                    label=sclass,
                )

                plt.ylim(plt.ylim())
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)

                if np.isfinite(conns_mn):
                    plt.plot(
                        conns_mn,
                        bar_pos[sidx] * max(plt.ylim()),
                        "o",
                        color=mean_colors[sidx],
                        markersize=3,
                    )
                    plt.plot(
                        [conns_mn - conns_sd, conns_mn + conns_sd],
                        np.full(2, bar_pos[sidx] * max(plt.ylim())),
                        "|-",
                        color=mean_colors[sidx],
                        linewidth=1.0,
                    )
                    plt.text(
                        conns_mn,
                        (bar_pos[sidx] + 0.01) * max(plt.ylim()),
                        f"{conns_mn:.2f}",
                        color=mean_colors[sidx],
                        va="bottom",
                        ha="center",
                    )

                plt.xlim([-10, 50])

                if lidx == 0:
                    plt.title(region)
                if lidx < len(layers) - 1:
                    plt.gca().set_xticklabels([])
                else:
                    plt.xlabel("#Syn/conn")

                if ridx == 0:
                    plt.ylabel(layer)

                if lidx == len(layers) - 1 and ridx == len(regions) - 1:
                    plt.legend()

            plt.suptitle(f"Thalamo-cortical projections [{sclass}]")

            plt.savefig(folder / f"synapses_per_connection_{sclass.lower()}.png", dpi=150)


def _voxel_based_densities(atlas_regions, pruned):
    """Compute voxel-based densities"""
    syn_pos = pruned[["x", "y", "z"]]
    syn_atlas_idx = atlas_regions.positions_to_indices(syn_pos.values)

    vox_syn_count = np.zeros_like(atlas_regions.raw, dtype=int)
    idx, counts = np.unique(syn_atlas_idx, axis=0, return_counts=True)
    vox_syn_count[tuple(idx.T)] = counts

    return vox_syn_count / atlas_regions.voxel_volume


def _voxel_relative_height(height, layer_thickness):
    """Compute voxel-based relative height values"""
    heights = relative_height_to_absolute(height.raw, layer_thickness)
    total_height = np.array(layer_thickness)[:, 1].astype(float).sum()
    return heights / total_height


def _atlas_ids(atlas, regions, layer_thickness):
    """Get the atlas IDs of the regions for each layer."""
    atlas_ids = np.zeros((len(layer_thickness), len(regions)), dtype=int)
    for ridx, region in enumerate(regions):
        for lidx, [layer, _] in enumerate(layer_thickness):
            atlas_id = get_region_ids(atlas, [str(layer)], [region])
            atlas_ids[lidx, ridx] = atlas_id[0]

    return atlas_ids


def _height_histogram(atlas, height, pruned, regions, layer_thickness):
    """Get the height histogram for each region."""
    # pylint: disable=too-many-locals
    atlas_regions = atlas.load_data("brain_regions")
    vox_syn_density = _voxel_based_densities(atlas_regions, pruned)
    voxel_height = _voxel_relative_height(height, layer_thickness)
    atlas_ids = _atlas_ids(atlas, regions, layer_thickness)

    # Take number of bins as the number of voxel-layers (along y-axis) that have synapses.
    # This is done to ensure there are no histograms "between" the voxels (ending up in nan values).
    n_bins = np.isfinite(height.raw).any(axis=2).any(axis=0).sum()
    bins = np.linspace(np.nanmin(voxel_height), np.nanmax(voxel_height), n_bins + 1)
    bin_centers = np.array([np.mean(bins[i : i + 2]) for i in range(n_bins)])

    rel_height_values = voxel_height[~np.isnan(voxel_height)]
    density_values = vox_syn_density[~np.isnan(voxel_height)]
    regid_values = atlas_regions.raw[~np.isnan(voxel_height)]

    # Compute height histogram per region
    density_hist = np.zeros((n_bins, len(regions)))
    for ridx in range(len(regions)):
        for didx in range(n_bins):
            dmin = bins[didx]
            dmax = bins[didx + 1]

            # To include border values in the last bin
            if didx + 1 == n_bins:
                dmax += 1

            # pylint: disable=assignment-from-no-return
            dsel = np.logical_and(rel_height_values >= dmin, rel_height_values < dmax)
            # pylint: enable=assignment-from-no-return
            rsel = np.in1d(regid_values, atlas_ids[:, ridx])

            # Mean density in a given height range
            density_hist[didx, ridx] = np.mean(density_values[np.logical_and(dsel, rsel)])

    # Estimate rel. layer height boundaries (voxel-based)
    layer_height_range = np.zeros((len(layer_thickness), len(regions), 2))
    for ridx in range(len(regions)):
        for lidx in range(len(layer_thickness)):
            dval_tmp = rel_height_values[regid_values == atlas_ids[lidx, ridx]]
            layer_height_range[lidx, ridx, :] = [np.min(dval_tmp), np.max(dval_tmp)]

    return density_hist, bins, bin_centers, layer_height_range


def synapse_density_profiles_region(
    atlas, height, pruned, density_params, regions, layer_thickness, folder
):
    """Plot expected and resulting density profile for each region."""
    # pylint: disable=too-many-locals

    # Rel. height profiles (voxel-based densities & height estimates)
    histogram, bins, bin_centers, height_range = _height_histogram(
        atlas, height, pruned, regions, layer_thickness
    )
    areas = []

    # Plot rel. synapse density profiles (voxel-based)
    lcolors = plt.cm.jet(np.linspace(0, 1, len(layer_thickness)))  # pylint: disable=no-member
    plt.figure(figsize=(8, 6))
    plt.gcf().patch.set_facecolor("w")
    for ridx, reg in enumerate(regions):
        plt.subplot(1, len(regions), ridx + 1)
        plt.barh(100.0 * bin_centers, histogram[:, ridx], np.diff(100.0 * bin_centers[:2]))
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.ylim([100.0 * bins[0], 100.0 * bins[-1]])
        plt.title(reg)
        plt.xlabel("Density\n[#Syn/um3]")

        if ridx == 0:
            plt.ylabel("Rel. height [%]")
        else:
            plt.gca().set_yticklabels([])

        plt.tight_layout()

        area_expected = 0

        # Plot intended densities in the recipe
        for ditem in density_params:
            dprof_steps_x, dprof_steps_y = intended_densities(
                ditem, ridx, height_range, len(layer_thickness)
            )
            area_expected += np.sum((dprof_steps_y[1:] - dprof_steps_y[:-1]) * dprof_steps_x[:-1])
            h_step = plt.step(
                dprof_steps_x, 100.0 * dprof_steps_y, "m", where="pre", linewidth=1.5, alpha=0.5
            )

        area = np.sum(histogram[:, ridx] * (bin_centers[1] - bin_centers[0]))
        areas.append([reg, area, area_expected])

        # Plot (overlapping) layer boundaries
        xlim = plt.xlim()
        xmax = max(xlim)
        for lidx, [lay, _] in enumerate(layer_thickness):
            plt.plot(
                np.ones(2) * xmax,
                100.0 * height_range[lidx, ridx, :],
                "-_",
                color=lcolors[lidx, :],
                linewidth=5,
                alpha=0.5,
                solid_capstyle="butt",
                markersize=10,
                clip_on=False,
            )

            plt.text(
                xmax,
                100.0 * np.mean(height_range[lidx, ridx, :]),
                f"  {lay}",
                color=lcolors[lidx, :],
                ha="left",
                va="center",
            )

            plt.plot(
                xlim,
                np.ones(2) * 100.0 * height_range[lidx, ridx, 0],
                "-",
                color=lcolors[lidx, :],
                linewidth=1,
                alpha=0.1,
                zorder=0,
            )

            plt.plot(
                xlim,
                np.ones(2) * 100.0 * height_range[lidx, ridx, 1],
                "-",
                color=lcolors[lidx, :],
                linewidth=1,
                alpha=0.1,
                zorder=0,
            )

    plt.legend(h_step, ["Recipe"], loc="lower right")
    plt.tight_layout()
    plt.savefig(folder / "synapse_density_profiles_voxel.png", dpi=150)

    return pd.DataFrame(areas, columns=["region", "area", "expected"])


def intended_densities(ditem, ridx, layer_height_range, n_layers):
    """Get the intended density for a density profile."""
    dens = np.array(ditem)

    # Translate relative notation to layer and its fraction: 2.45 > 2, .45
    layers, fractions = (dens[:, 0] // 1).astype(int), dens[:, 0] % 1

    # Fix the issue with last layer having fraction of 1.0 giving 'extra' layer
    # TODO: figure out how to best mitigate this issue
    fractions[layers == n_layers] = 1.0
    layers[layers == n_layers] -= 1

    ranges = layer_height_range[layers, ridx, :]
    diffs = np.diff(ranges).flatten()
    densities = dens[:, 1]
    steps = ranges[:, 0] + diffs * fractions

    return densities, steps


def fiber_coverage(pruned, fibers, folder):
    """Plots fiber usage as a scatter plot"""
    selected = np.in1d(fibers.index.to_numpy(), pruned.sgid.unique())
    fsel = fibers.loc[selected]
    fnon = fibers.loc[np.invert(selected)]
    _, ax = _get_ax()
    plt.scatter(fsel["x"], fsel["z"], c="g", s=1)
    plt.scatter(fnon["x"], fnon["z"], c="r", s=1)
    ax.axis("equal")

    plt.savefig(folder / "fiber_coverage.png", dpi=150)

    return sum(selected) / len(selected)


def create_report(coverage, density, folder):
    """Create a `summary.txt` of the projections."""

    result = "SUCCESS"
    messages = []
    density_overall = density.area.sum() / density.expected.sum()
    report = "Projetionizer Report\n"
    report += "--------------------\n"
    report += f"Fiber coverage: {coverage}\n"
    report += f"Overall density: {density_overall}\n"
    report += "Density per region:\n"

    for _, r in density.iterrows():
        report += f"- {r.region}: {r.area/r.expected}\n"

    report += "--------------------\n"

    if np.abs(density_overall - 1) > DENSITY_THRESHOLD:
        msg = f"Overall synapse density deviates from the profile by more than {DENSITY_THRESHOLD}."
        L.warning(msg)
        messages.append(msg)
        result = "FAIL"

    if coverage < FIBER_THRESHOLD:
        msg = f"Fiber coverage is less than {DENSITY_THRESHOLD}"
        L.warning(msg)
        messages.append(msg)
        result = "FAIL"

    report += f"Result: {result}\n"
    for m in messages:
        report += f"- {m}\n"

    (folder / "report.txt").write_text(report)


class LayerThickness(JsonTask):
    """Generate json data containing the mean layer thicknesses for each layer."""

    def run(self):  # pragma: no cover
        res = []
        atlas = Circuit(self.circuit_config).atlas
        for layer in self.layers:  # pylint: disable=not-an-iterable
            ph = atlas.load_data(f"[PH]{convert_layer_to_PH_format(layer)}")
            thickness = ph.raw[..., 1] - ph.raw[..., 0]
            mean = thickness[np.isfinite(thickness)].mean()
            res.append([layer, float(mean)])

        with self.output().open("w") as outfile:
            json.dump(res, outfile)


class Analyse(CommonParams):
    """Run the analysis"""

    def requires(self):  # pragma: no cover
        return (
            self.clone(ReducePrune),
            self.clone(FullSample),
            self.clone(ChooseConnectionsToKeep),
            self.clone(CutoffMeans),
            self.clone(SynapseDensity),
            self.clone(VirtualFibers),
            self.clone(Height),
            self.clone(LayerThickness),
        )

    def run(self):  # pragma: no cover
        # pylint: disable=too-many-locals
        (
            pruned,
            sampled,
            connections,
            cutoffs,
            distmap,
            fibers,
            height,
            layer_thickness,
        ) = load_all(self.input())

        regions = self.get_regions()
        atlas = Circuit(self.circuit_config).atlas

        pruned_no_edge = remove_synapses_with_sgid(pruned, fibers[fibers["apron"]].index)

        fraction_pruned_vs_height(self.folder, self.n_total_chunks)
        innervation_width(pruned, self.circuit_config, self.folder)

        synapse_density_per_voxel(
            self.folder, sampled, layer_thickness, distmap, self.oversampling, "sampled"
        )
        synapse_density_per_voxel(
            self.folder, pruned_no_edge, layer_thickness, distmap, self.oversampling, "pruned"
        )
        synapse_density(pruned_no_edge, distmap, layer_thickness, folder=self.folder)
        density_areas = synapse_density_profiles_region(
            atlas, height, pruned, distmap, regions, layer_thickness, self.folder
        )
        synapse_heights(pruned_no_edge, atlas, folder=self.folder)

        thalamo_cortical_cells_per_fiber(pruned, self.folder)
        distribution_synapses_per_connection(pruned, self.folder)
        distribution_synapses_per_connection_per_layer(
            pruned, regions, self.layers, self.circuit_config, self.folder
        )
        syns_per_connection(connections, cutoffs, self.folder, self.target_mtypes)

        efferent_neuron_per_fiber(pruned, fibers, self.folder)

        coverage = fiber_coverage(pruned, fibers, self.folder)

        create_report(coverage, density_areas, self.folder)

        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)
