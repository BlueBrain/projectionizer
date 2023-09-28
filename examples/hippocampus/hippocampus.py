#!/usr/bin/env python
"""Hippocampus projections."""
import logging
import shutil
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from bluepy import Circuit

from projectionizer import (
    afferent_section_position,
    hippocampus,
    synapses,
    utils,
    version,
)
from projectionizer import write_sonata as sonata

L = logging.getLogger(__name__)

ASSIGN_PATH = "ASSIGNMENT"
MERGE_PATH = "MERGE"
SONATA_PATH = "SONATA"
SAMPLE_PATH = "SAMPLED"
CHUNK_SIZE = 100000000

ASSIGNED_COLS = [
    "sgid",
    "tgid",
    "section_id",
    "segment_id",
    "synapse_offset",
    "section_pos",
    "section_type",
    "x",
    "y",
    "z",
]

REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)


class DataFrameWrapper:
    """Wrapper for the separately saved pandas.DataFrame columns.

    To have the separate columns to act as if they were in same DataFrame (for write_sonata).
    """

    def __init__(self, path, distance_soma):
        self._path = Path(path)
        self._length = None
        self._distance_soma = distance_soma

    @property
    def sgid_path_distance(self):
        return pd.Series(
            np.full(len(self), self._distance_soma), name="sgid_path_distance", dtype=np.float32
        )

    @property
    def columns(self):
        return ASSIGNED_COLS

    def __len__(self):
        if self._length is None:
            self._length = len(self[ASSIGNED_COLS[0]])

        return self._length

    def _get_column_path(self, column):
        return (self._path / column).with_suffix(".feather")

    def __getattr__(self, attribute):
        if attribute in self.columns:
            return pd.read_feather(self._get_column_path(attribute))[attribute]
        return self.__getattribute__(attribute)

    def __getitem__(self, item):
        return self.__getattr__(item)


def _path_exists(path):
    exists = path.exists()
    if exists:
        L.info("Already have %s, skipping", path)

    return exists


@click.group()
@click.version_option(version.VERSION)
@click.option("-c", "--config", type=REQUIRED_PATH, required=True)
@click.option("-o", "--output", required=True)
@click.option("-v", "--verbose", count=True)
@click.pass_context
def cli(ctx, config, verbose, output):
    """Hippocampus projections generation"""
    L.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)])
    if verbose > 3:
        import multiprocessing.util  # pylint: disable=import-outside-toplevel

        multiprocessing.util.log_to_stderr(logging.DEBUG)

    output, config = Path(output), Path(config)
    output.mkdir(parents=True, exist_ok=True)
    orig_config = output / config.name
    if orig_config.exists():
        orig_config_contents = utils.load(orig_config)
        config_contents = utils.load(config)

        if config_contents != orig_config_contents:
            sys.exit(
                f"Config at {orig_config} already exists, and is different. "
                "Please delete it or pick another directory"
            )
    else:
        # str() can be removed in python >= 3.8
        shutil.copy(str(config), str(output))

    ctx.obj["config"] = utils.load(config)

    ctx.obj["output"] = output


@cli.command()
@click.option("--region", required=True)
@click.pass_context
def full_sample(ctx, region):
    """fully sample all the regions"""
    config, output = ctx.obj["config"], ctx.obj["output"]
    output = output / SAMPLE_PATH
    output.mkdir(parents=True, exist_ok=True)

    atlas = Circuit(config["circuit_config"]).atlas
    index_path = config["spatial_index_path"]

    assert region in config["region_percentages"]

    region_map = atlas.load_region_map()
    brain_regions = atlas.load_data("brain_regions")

    region_ids = region_map.find("@" + region, attr="acronym")

    for id_ in region_ids:
        L.debug("Sampling %s[%s]", region, id_)
        hippocampus.full_sample_parallel(brain_regions, region, id_, index_path, output)


def _assign_sgid(syns, sgid_start, sgid_count):
    """assign source gids to syns"""
    if sgid_count == 1:
        syns["sgid"] = sgid_start
    else:
        sgids = np.arange(sgid_start, sgid_start + sgid_count)
        syns["sgid"] = np.random.choice(sgids, size=len(syns), replace=True)

    return syns


def _assign(cells, morph_class, count, output, region, sgid_start, sgid_count, config):
    samples_path = output / SAMPLE_PATH
    output = output / ASSIGN_PATH
    output.mkdir(parents=True, exist_ok=True)

    L.debug("Assign doing: %s[%s] with %d synapses", region, morph_class, count)

    segs = [utils.read_feather(path) for path in samples_path.glob(f"{region}_*.feather")]
    segs = pd.concat(segs, sort=False, ignore_index=True).join(cells["morph_class"], on="tgid")
    segs = segs[segs["morph_class"] == morph_class].drop(["morph_class"], axis=1)

    if len(segs) == 0:
        return

    for i, size in enumerate([CHUNK_SIZE] * (count // CHUNK_SIZE) + [(count % CHUNK_SIZE)]):
        path = output / f"{region}_{morph_class}_{i:05d}.feather"
        if not _path_exists(path):
            with utils.delete_file_on_exception(path):
                syns = synapses.pick_synapse_locations(segs, synapses.segment_pref_length, size)
                L.debug("Subsample: %s", len(syns))
                syns.drop(
                    synapses.SEGMENT_START_COLS + synapses.SEGMENT_END_COLS,
                    axis="columns",
                    inplace=True,
                )
                syns = _assign_sgid(syns, sgid_start, sgid_count)
                morphs = utils.get_morphs_for_nodes(
                    config["target_node_path"],
                    config["target_node_population"],
                    config["morphology_path"],
                    config["morphology_type"],
                )
                syns["section_pos"] = afferent_section_position.compute_positions(syns, morphs)
                utils.write_feather(path, syns)


@cli.command()
@click.option("--region", required=True)
@click.pass_context
def assign(ctx, region):
    """create correct count of synapses, assign their sgid

    This is performed separately for `INT` and `PYR` synapses
    """
    config, output = ctx.obj["config"], ctx.obj["output"]

    cells = Circuit(config["circuit_config"]).cells.get()

    synapse_count = config["synapse_count_per_type"]
    int_count = int(
        config["region_percentages"][region]
        * synapse_count["INT"]
        * np.count_nonzero(cells.morph_class == "INT")
    )
    pyr_count = int(
        config["region_percentages"][region]
        * synapse_count["PYR"]
        * np.count_nonzero(cells.morph_class == "PYR")
    )

    sgid_start, sgid_count = config["sgid_start"], config["sgid_count"]

    _assign(cells, "INT", int_count, output, region, sgid_start, sgid_count, config)
    _assign(cells, "PYR", pyr_count, output, region, sgid_start, sgid_count, config)


@cli.command()
@click.pass_context
def merge(ctx):
    """merge the data in the assigned files column by column (one file per column)"""
    output = ctx.obj["output"]
    assign_paths = [*(output / ASSIGN_PATH).glob("*.feather")]

    output = output / MERGE_PATH
    output.mkdir(parents=True, exist_ok=True)

    for col in ASSIGNED_COLS:
        filepath = output / f"{col}.feather"
        if not _path_exists(filepath):
            L.debug("Merging column %s", col)
            data = pd.concat(
                (utils.read_feather(f, columns=[col]) for f in assign_paths),
                sort=False,
                ignore_index=False,
            )

            L.debug("Writing %s", filepath)
            with utils.delete_file_on_exception(filepath):
                utils.write_feather(filepath, data)


@cli.command()
@click.pass_context
def write_sonata(ctx):
    """Write out the h5 synapse files (for nodes and edges), corresponding to the assignment"""
    config, output = ctx.obj["config"], ctx.obj["output"]
    column_path = output / MERGE_PATH

    output = output / SONATA_PATH
    output.mkdir(parents=True, exist_ok=True)

    syns = DataFrameWrapper(column_path, config["distance_soma"])
    node_path = output / "nodes.h5"
    edge_path = output / "nonparameterized-edges.h5"

    if not _path_exists(node_path):
        node_population = config.get("node_population_name", "projections")

        L.debug("Writing %s", node_path)
        with utils.delete_file_on_exception(node_path):
            sonata.write_nodes(syns, node_path, node_population, "virtual")

    if not _path_exists(edge_path):
        edge_population = config.get("edge_population_name", "projections")

        L.debug("Writing %s", edge_path)
        with utils.delete_file_on_exception(edge_path):
            sonata.write_edges(syns, edge_path, edge_population)


if __name__ == "__main__":
    cli(obj={})  # pylint: disable=no-value-for-parameter
