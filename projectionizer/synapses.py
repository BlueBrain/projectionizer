"""Functions for picking synpases"""
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
from bluepy import Section, Segment
from bluepy.index import SegmentIndex
from neurom import NeuriteType
from tqdm import tqdm

from projectionizer.utils import (
    SECTION_TYPE_MAP,
    ErrorCloseToZero,
    convert_to_smallest_allowed_int_type,
    in_bounding_box,
    map_parallelize,
    min_max_axis,
    normalize_probability,
)

L = logging.getLogger(__name__)

SEGMENT_START_COLS = [
    Segment.X1,
    Segment.Y1,
    Segment.Z1,
]
SEGMENT_END_COLS = [
    Segment.X2,
    Segment.Y2,
    Segment.Z2,
]

WANTED_COLS = [
    "gid",
    Section.ID,
    Segment.ID,
    "segment_length",
    "synapse_offset",
    "section_type",
    "x",
    "y",
    "z",
]
XYZ = list("xyz")
SOURCE_XYZ = ["source_x", "source_y", "source_z"]
VOLUME_TRANSMISSION_COLS = ["sgid"] + WANTED_COLS + SOURCE_XYZ + ["distance_volume_transmission"]
INT_COLS = ["gid", Section.ID, Segment.ID]


def segment_pref_length(df):
    """don't want axons, assign probability of 0 to them, and 1 to other neurite types,
    multiplied by the length of the segment
    this will be normalized by the caller
    """
    return df["segment_length"] * (df[Section.NEURITE_TYPE] != NeuriteType.axon).astype(float)


def build_synapses_default(height, synapse_density, oversampling):
    """Build voxel count from densities according to the height along the column.

    Height and densities can be in relative height <layer_index>.<fraction> format or absolute
    values. (See: sscx.recipe_to_relative_height_and_density)
    """
    raw = np.zeros_like(height.raw, dtype=np.uint)  # pylint: disable=no-member

    voxel_volume = np.prod(np.abs(height.voxel_dimensions))
    for dist in synapse_density:
        for (bottom, density), (top, _) in zip(dist[:-1], dist[1:]):
            with np.errstate(invalid="ignore"):  # ignore warning about nans in height.raw
                idx = (bottom <= height.raw) & (height.raw < top)
            idx = np.nonzero(np.nan_to_num(idx, 0))
            count = voxel_volume * density * oversampling
            if count < 1:
                datalen = np.shape(idx)[1]
                raw[idx] = (count > np.random.rand(datalen)).astype(int)
                L.debug("assigned density: %.3f, target:%.3f", np.sum(raw[idx]) / datalen, count)
            else:
                raw[idx] = int(count)

    return height.with_data(raw)


def _sample_with_flat_index(index_path, min_xyz, max_xyz):  # pragma: no cover
    """use flat index to get segments within min_xyz, max_xyz"""
    #  this is loaded late so that multiprocessing loads it outside of the main
    #  python binary - at one point, this was necessary, as there was shared state
    import libFLATIndex as FI  # pylint:disable=import-outside-toplevel

    try:
        index = FI.loadIndex(str(os.path.join(index_path, "SEGMENT")))  # pylint: disable=no-member
        min_xyz_ = tuple(map(float, min_xyz))
        max_xyz_ = tuple(map(float, max_xyz))
        segs_df = FI.numpy_windowQuery(index, *(min_xyz_ + max_xyz_))  # pylint: disable=no-member
        segs_df = SegmentIndex._wrap_result(segs_df)  # pylint: disable=protected-access
        FI.unLoadIndex(index)  # pylint: disable=no-member
        del index
    except Exception:  # pylint: disable=broad-except
        return None

    return segs_df.sort_values(segs_df.columns.tolist())


def get_segment_limits_within_sphere(starts, ends, pos, radius):
    """Computes segments' start and end points within a sphere

    Args:
        starts(np.array): Mx3 array giving the segment start points (X,Y,Z)
        ends(np.array): Mx3 array giving the segment end points (X,Y,Z)
        pos(np.array): center point of the sphere (X,Y,Z)
        radius(float): radius of the sphere

    Returns:
        start_point(np.array): start points of the segments within the sphere
        end_point(np.array): end points of the segments within the sphere
    """
    # pylint: disable=too-many-locals
    starts = starts.copy()
    ends = ends.copy()

    segment = ends - starts
    direction = segment / np.linalg.norm(segment, axis=1)[:, None]
    start_to_pos = pos - starts
    end_to_pos = pos - ends
    magnitude_start_to_pos = np.sum(start_to_pos * direction, axis=1)
    magnitude_end_to_pos = np.sum(end_to_pos * direction, axis=1)

    # Segments for which the closest point is within radius but the segment is out.
    # I.e., both the start and end points are in the same direction from the closest point.
    start_mask = np.sum(start_to_pos**2, axis=1) > radius**2
    end_mask = np.sum(end_to_pos**2, axis=1) > radius**2
    segment_mask = np.logical_and(
        np.logical_and(start_mask, end_mask),
        np.sign(magnitude_start_to_pos) == np.sign(magnitude_end_to_pos),
    )
    starts[segment_mask] = np.nan
    ends[segment_mask] = np.nan

    # the radius is the hypothenuse of a triangle defined by three points:
    # center of the sphere, point where the line (on which the segment lies) is closest to the
    # center and the point where the line intersects with the surface
    closest_point = magnitude_start_to_pos[:, None] * direction + starts
    shortest_distance_squared = np.sum((closest_point - pos) ** 2, axis=1)
    with np.errstate(invalid="ignore"):  # ignore warning of negative values (will result in nan)
        distance_to_surface = np.sqrt(radius**2 - shortest_distance_squared)

    # If start/end point is outside radius but segment is in, replace with the surface point
    start_mask &= ~segment_mask
    end_mask &= ~segment_mask
    point_to_surface = distance_to_surface[:, None] * direction
    starts[start_mask] = closest_point[start_mask] - point_to_surface[start_mask]
    ends[end_mask] = closest_point[end_mask] + point_to_surface[end_mask]

    return starts, ends


def spherical_sampling(pos_sgid, index_path, radius):
    """Get segments within a sphere with a given position and radius

    Args:
        pos_sgid(tuple): a tuple with XYZ-position of a synapse and its sgid
        index_path(str): path to the circuit directory
        radius(float): maximum radius of the volume_transmission
    """
    position, sgid = pos_sgid
    min_xyz = position - radius
    max_xyz = position + radius

    segs_df = _sample_with_flat_index(index_path, min_xyz, max_xyz)
    starts = segs_df[SEGMENT_START_COLS].to_numpy().astype(float)
    ends = segs_df[SEGMENT_END_COLS].to_numpy().astype(float)

    mask_nonzero_length = np.any(starts != ends, axis=1)
    segs_df = segs_df[mask_nonzero_length]
    starts = starts[mask_nonzero_length]
    ends = ends[mask_nonzero_length]

    # Assuming the segment is a line (no R1, R2)
    # https://bbpteam.epfl.ch/project/issues/browse/NSETM-1482#comment-153675
    segment_start, segment_end = get_segment_limits_within_sphere(starts, ends, position, radius)

    # NOTE by herttuai on 09/06/2021:
    # Storing WANTED_COLS, source position, sgid and distance
    # Probably better to store the index in the reduce-prune.feather than the original position.
    alpha = np.random.random(segment_start.shape[0])
    synapse = alpha[:, None] * (segment_end - segment_start) + segment_start
    segs_df[XYZ] = synapse
    segs_df[SOURCE_XYZ] = position
    segs_df.loc[:, "sgid"] = sgid
    segs_df.loc[:, "synapse_offset"] = np.linalg.norm(synapse - starts, axis=1)
    segs_df.loc[:, "segment_length"] = np.linalg.norm(ends - starts, axis=1)
    segs_df.loc[:, "distance_volume_transmission"] = np.linalg.norm(synapse - position, axis=1)
    segs_df.loc[:, "section_type"] = np.fromiter(
        (SECTION_TYPE_MAP[x] for x in segs_df[Section.NEURITE_TYPE]),
        dtype=np.int16,
        count=len(segs_df.index),
    )

    return segs_df[VOLUME_TRANSMISSION_COLS].dropna()


def pick_synapses_voxel(xyz_counts, index_path, segment_pref, dataframe_cleanup):
    """Select `count` synapses from the circuit that lie between `min_xyz` and `max_xyz`

    Args:
        xyz_counts(tuple of min_xyz, max_xyz, count): bounding box and count of synapses desired
        index_path(str): absolute path to circuit path, where a SEGMENT exists
        segment_pref(callable (df -> floats)): function to assign probabilities per segment
        dataframe_cleanup(callable (df -> df)): function to remove any unnecessary columns
        and do other processing, *must do all operations in place*, None if not needed

    Returns:
        DataFrame with `WANTED_COLS`
    """
    min_xyz, max_xyz, count = xyz_counts

    segs_df = _sample_with_flat_index(index_path, min_xyz, max_xyz)

    if segs_df is None:
        return None

    # pylint: disable=unsubscriptable-object
    starts = segs_df[SEGMENT_START_COLS].to_numpy().astype(float)
    ends = segs_df[SEGMENT_END_COLS].to_numpy().astype(float)
    # pylint: enable=unsubscriptable-object

    # keep only the segments whose midpoints are in the current voxel
    in_bb = pd.DataFrame((ends + starts) / 2.0, columns=list("xyz"), index=segs_df.index)
    in_bb = in_bounding_box(*min_max_axis(min_xyz, max_xyz), df=in_bb)

    segs_df = segs_df[in_bb].copy()  # pylint: disable=unsubscriptable-object

    if len(segs_df) == 0:
        return None

    segs_df["segment_length"] = np.linalg.norm(ends[in_bb] - starts[in_bb], axis=1)
    segs_df["segment_length"] = segs_df["segment_length"].astype(np.float32)

    prob_density = segment_pref(segs_df)
    try:
        prob_density = normalize_probability(prob_density)
    except ErrorCloseToZero:
        return None

    picked = np.random.choice(np.arange(len(segs_df)), size=count, replace=True, p=prob_density)
    segs_df = segs_df.iloc[picked].reset_index()

    alpha = np.random.random(size=len(segs_df))

    segs_df["synapse_offset"] = alpha * segs_df["segment_length"]
    segs_df["section_type"] = np.array(
        [SECTION_TYPE_MAP[x] for x in segs_df[Section.NEURITE_TYPE]], dtype=np.int16
    )

    segs_df = segs_df.join(
        pd.DataFrame(
            alpha[:, None] * segs_df[SEGMENT_START_COLS].to_numpy().astype(float)
            + (1.0 - alpha[:, None]) * segs_df[SEGMENT_END_COLS].to_numpy().astype(float),
            columns=list("xyz"),
            dtype=np.float32,
            index=segs_df.index,
        )
    )

    segs_df["synapse_offset"] = alpha * segs_df["segment_length"]
    segs_df = segs_df[WANTED_COLS]

    if dataframe_cleanup is not None:
        dataframe_cleanup(segs_df)

    return segs_df


def downcast_int_columns(df):
    """Downcast int columns"""
    for name in INT_COLS:
        df[name] = convert_to_smallest_allowed_int_type(df[name])


def pick_synapses(
    index_path,
    synapse_counts,
    segment_pref=segment_pref_length,
    dataframe_cleanup=downcast_int_columns,
):
    """Sample segments from circuit
    Args:
        index_path: absolute path to circuit path, where a SEGMENT exists
        synapse_counts(VoxelData):
            A VoxelData containing the number of segment to be sampled in each voxel

    Returns:
        a DataFrame with the following columns:
            ['tgid', 'Section.ID', 'Segment.ID', 'segment_length', 'x', 'y', 'z']
    """

    idx = np.nonzero(synapse_counts.raw)

    min_xyzs = synapse_counts.indices_to_positions(np.transpose(idx))
    max_xyzs = min_xyzs + synapse_counts.voxel_dimensions
    xyz_counts = zip(min_xyzs, max_xyzs, synapse_counts.raw[idx])

    func = partial(
        pick_synapses_voxel,
        index_path=index_path,
        segment_pref=segment_pref,
        dataframe_cleanup=dataframe_cleanup,
    )

    synapses = list(map_parallelize(func, tqdm(xyz_counts, total=len(min_xyzs))))

    n_none_dfs = sum(df is None for df in synapses)
    percentage_none = n_none_dfs / float(len(synapses)) * 100
    if percentage_none > 20.0:  # pragma: no cover
        L.warning("%s of dataframes are None.", percentage_none)

    L.debug("Picking finished. Now concatenating...")
    return pd.concat(synapses, ignore_index=True)


def organize_indices(synapses):
    """*inplace* reorganize the synapses indices"""
    synapses.set_index(["tgid", "sgid"], inplace=True)
    synapses.sort_index(inplace=True)
    synapses.reset_index(inplace=True)

    return synapses
