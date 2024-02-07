"""Functions to compute afferent section positions for the synapses"""

import logging
from functools import partial

import numpy as np
import pandas as pd
from morphio import Morphology
from tqdm import tqdm

from projectionizer.utils import map_parallelize

L = logging.getLogger(__name__)

# Threshold for computed afferent_section_pos above which a warning is printed
WARNING_THRESHOLD = 1.001


def get_morph_section(morph, section_id_sonata):
    """Helper function to get section of morphology with given ID.

    To get translate SONATA to MorphIO indexing, 1 is subtracted from the section id, since in
    MorphIO, soma is its own property and section indexing starts at 0. In SONATA, soma is referred
    to with a section index of 0.

    This function exists to ensure the section is always acquired in the same manner and ease the
    transition in case the indexing is ever changed either in SONATA or in MorphIO.

    Args:
        morph (morphio.Morphology): Morphology instance.
        section_id_sonata (int): Section ID in SONATA (i.e., soma==0) format.

    Returns:
        morphio.Section: Desired Section instance.
    """
    return morph.section(section_id_sonata - 1)


def compute_afferent_section_pos(row, morph):
    """Computes the afferent_section_pos

    Args:
        row (namedtuple): row of the synapse dataframe containing the orig_index column
        morph(morphio.Morphology): A morphio immutable morphology

    Returns:
        float: the afferent section position for the row entry
    """
    section = get_morph_section(morph, row.section_id)
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


def compute_positions_worker(morph_df):
    """Worker function computing the afferent_section_pos values for all the sgids that connect to
    any of the nodes with the given morphology.

    Args:
        morph_df(tuple): tuple with morph path and its entries in the synapses dataframe

    Returns:
        tuple: the section_pos entries for a morphology and the indices in the synapses dataframe
    """
    morph_path, df = morph_df
    morph = Morphology(morph_path)
    func = partial(compute_afferent_section_pos, morph=morph)
    positions = np.fromiter(map(func, df.itertuples()), dtype=np.float32)

    return positions, df.orig_index.to_numpy()


def compute_positions(synapses, morphs):
    """Parallelizes afferent section positions for the entered synapses.

    Args:
        synapses(pandas.DataFrame): synapse dataframe (e.g., from step_2_prune.ReducePrune)
        morphs(pandas.DataFrame): dataframe containing nodes' absolute morph paths

    Returns:
        pandas.DataFrame: the afferent section positions for the synapses
    """
    # Specify needed columns to minimize memory footprint
    columns = ["tgid", "section_id", "segment_id", "synapse_offset"]
    syns = synapses[columns].reset_index().rename(columns={"index": "orig_index"})

    # Fetching morph names with `libsonata` in the worker slows down the parallel process
    # significantly. Getting the morphs here and then grouping by them later.
    syns = syns.join(morphs, on="tgid").drop("tgid", axis="columns")

    L.info("Computing afferent section positions...")
    ret = map_parallelize(
        compute_positions_worker,
        tqdm(syns.groupby("morph")),
        maxtasksperchild=None,
    )

    L.info("Concatenating and sorting the results...")
    ret, idx = np.concatenate(ret, axis=1)
    ret[idx.astype(int)] = np.copy(ret)  # needs to be a copy!

    return pd.DataFrame(ret.astype(np.float32), columns=["section_pos"])
