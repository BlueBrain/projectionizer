"""Functions to compute afferent section positions for the synapses"""
import logging
from functools import partial
from pathlib import Path

import libsonata
import numpy as np
import pandas as pd
from morphio import Morphology
from tqdm import tqdm

from projectionizer.utils import map_parallelize

L = logging.getLogger(__name__)

# Threshold for computed afferent_section_pos above which a warning is printed
WARNING_THRESHOLD = 1.001


def load_morphology(morph_name, morph_path, morph_type):
    """Load the given morph from the path based on the morph_type.

    Args:
        morhp_name(str): the name of the morphology
        morph_path(str): path to the directory containing the morphology
        morph_type(str): the morphology type (h5, asc, swc)

    Returns:
        morphio.Morphology: A morphio immutable morphology.
    """
    filename = f"{morph_name}.{morph_type}"

    return Morphology(Path(morph_path, filename))


def compute_afferent_section_pos(row, morph):
    """Computes the afferent_section_pos

    Args:
        row (namedtuples): row of the synapse dataframe containing the orig_index column
        morph(morphio.Morphology): A morphio immutable morphology

    Returns:
        float: the afferent section position for the row entry
    """
    # Subtract 1 from the section id, since in morphio, soma is its own property and section
    # indexing starts at 0. In SONATA, soma is referred with an index of 0.
    section = morph.section(row.section_id - 1)
    segment_lengths = np.linalg.norm(np.diff(section.points, axis=0), axis=1)
    len_to_segment = segment_lengths[: row.segment_id].sum()
    computed_pos = (len_to_segment + row.synapse_offset) / segment_lengths.sum()

    if computed_pos > WARNING_THRESHOLD:
        L.warning(
            "Value exceeds threshold: %f, index in synapse dataframe: %d",
            computed_pos,
            row.orig_index,
        )

    return min(computed_pos, 1)


def compute_positions_worker(morph_df, morph_path, morph_type):
    """Worker function computing the afferent_section_pos values for all the sgids that connect to
    any of the nodes with the given morphology.

    Args:
        morph_df(tuple): tuple with morph name and its entries in the synapses dataframe
        morph_path(str): path to the directory containing the morphologies
        morph_type(str): the morphology type (h5, asc, swc)

    Returns:
        tuple: the section_pos entries for a morphology and the indices in the synapses dataframe
    """
    morph_name, df = morph_df
    morph = load_morphology(morph_name, morph_path, morph_type)
    func = partial(compute_afferent_section_pos, morph=morph)
    positions = np.fromiter(map(func, df.itertuples()), dtype=np.float32)

    return positions, df.orig_index.to_numpy()


def get_morphs_for_nodes(node_path, population):
    """Get morphology names for the nodes in a node file.

    Args:
        node_path(str): path to the nodes.h5 file containing the afferent nodes
        population(str): name of the node population

    Returns:
        pandas.DataFrame: the
    """
    node_storage = libsonata.NodeStorage(node_path)
    node_population = node_storage.open_population(population)
    morphs = node_population.get_attribute("morphology", node_population.select_all())

    # Sonata is 0-based so add the offset of 1 to the index to match the synapses.tgid
    return pd.DataFrame({"morph": morphs}, index=range(1, len(morphs) + 1))


def compute_positions(synapses, node_path, node_population, morph_path, morph_type):
    """Parallelizes afferent section positions for the entered synapses.

    Args:
        synapses(pandas.DataFrame): synapse dataframe (e.g., from step_2_prune.ReducePrune)
        node_path(str): path to the nodes.h5 file containing the afferent nodes
        node_population(str): name of the afferent node population
        morph_path(str): path to the directory containing the morphologies
        morph_type(str): the morphology type (h5, asc, swc)

    Returns:
        pandas.DataFrame: the afferent section positions for the synapses
    """
    # Specify needed columns to minimize memory footprint
    columns = ["tgid", "section_id", "segment_id", "synapse_offset"]
    syns = synapses[columns].reset_index().rename(columns={"index": "orig_index"})

    # Fetching morph names with libsonata in the worker slows down the parallel process
    # significantly. Getting the morphs here and then grouping by them later.
    morphs = get_morphs_for_nodes(node_path, node_population)
    syns = syns.join(morphs, on="tgid").drop("tgid", axis="columns")

    func = partial(
        compute_positions_worker,
        morph_path=morph_path,
        morph_type=morph_type,
    )

    L.info("Computing afferent section positions...")
    ret = map_parallelize(func, tqdm(syns.groupby("morph")), maxtasksperchild=None, chunksize=1)

    L.info("Concatenating and sorting the results...")
    ret, idx = np.concatenate(ret, axis=1)
    ret[idx.astype(int)] = np.copy(ret)  # needs to be a copy!

    return pd.DataFrame(ret.astype(np.float32), columns=["section_pos"])
