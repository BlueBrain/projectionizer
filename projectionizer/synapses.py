"""Functions for picking synapses"""

import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import spatial_index
import spatial_index.experimental
from morphio import SectionType
from tqdm import tqdm

from projectionizer.utils import (
    PARALLEL_JOBS,
    ErrorCloseToZero,
    convert_to_smallest_allowed_int_type,
    in_bounding_box,
    map_parallelize,
    min_max_axis,
    normalize_probability,
)

L = logging.getLogger(__name__)

SEGMENT_START_COLS = [
    "segment_x1",
    "segment_y1",
    "segment_z1",
]
SEGMENT_END_COLS = [
    "segment_x2",
    "segment_y2",
    "segment_z2",
]

WANTED_COLS = [
    "gid",
    "section_id",
    "segment_id",
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
INT_COLS = ["gid", "section_id", "segment_id"]
# per-process max cache size for spatial-index
CACHE_SIZE_MB = int(os.environ.get("SPATIAL_INDEX_CACHE_SIZE_MB", 4000))


def segment_pref_length(df):
    """don't want axons, assign probability of 0 to them, and 1 to other neurite types,
    multiplied by the length of the segment
    this will be normalized by the caller
    """
    return df["segment_length"] * (df["section_type"] != SectionType.axon).astype(float)


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


def _sample_with_spatial_index(index, min_xyz, max_xyz):  # pragma: no cover
    """use spatial index to get segments within min_xyz, max_xyz"""
    res = index.box_query(
        min_xyz, max_xyz, fields={"gid", "endpoints", "section_id", "segment_id", "section_type"}
    )
    starts, ends = res["endpoints"]
    del res["endpoints"]

    segs_df = pd.DataFrame(res)
    segs_df[SEGMENT_START_COLS] = starts
    segs_df[SEGMENT_END_COLS] = ends

    # `.dropna()` is used to get rid of somas (will be allowed in NSETM-2010)
    # `columns` need to be sorted to ensure the order of the segments is always the same.
    return segs_df.dropna().sort_values(sorted(segs_df.columns), ignore_index=True)


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

    # the radius is the hypotenuse of a triangle defined by three points:
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
        index_path(Path): path to the circuit directory
        radius(float): maximum radius of the volume_transmission
    """
    # TODO: check functionality that can be combined w/ pick_synapses_voxel
    # pylint: disable=too-many-locals
    position, sgid = pos_sgid
    min_xyz = position - radius
    max_xyz = position + radius

    segs_df = _sample_with_spatial_index(index_path, min_xyz, max_xyz)
    starts = segs_df[SEGMENT_START_COLS].to_numpy().astype(float)
    ends = segs_df[SEGMENT_END_COLS].to_numpy().astype(float)

    mask_nonzero_length = np.any(starts != ends, axis=1)

    # .copy() to avoid `pandas.core.common.SettingWithCopyWarning`
    segs_df = segs_df[mask_nonzero_length].copy()
    starts = starts[mask_nonzero_length]
    ends = ends[mask_nonzero_length]

    # Assuming the segment is a line (no R1, R2)
    # https://bbpteam.epfl.ch/project/issues/browse/NSETM-1482#comment-153675
    segment_start, segment_end = get_segment_limits_within_sphere(starts, ends, position, radius)

    # NOTE:
    # Storing WANTED_COLS, source position, sgid and distance
    # Probably better to store the index in the reduce-prune.feather than the original position.
    alpha = np.random.random(segment_start.shape[0])
    synapse = alpha[:, None] * (segment_end - segment_start) + segment_start
    segs_df[XYZ] = synapse
    segs_df[SOURCE_XYZ] = position
    segs_df["sgid"] = sgid
    segs_df["synapse_offset"] = np.linalg.norm(synapse - starts, axis=1)
    segs_df["segment_length"] = np.linalg.norm(ends - starts, axis=1)
    segs_df["distance_volume_transmission"] = np.linalg.norm(synapse - position, axis=1)

    return segs_df[VOLUME_TRANSMISSION_COLS].dropna()


def pick_segments_voxel(index, min_xyz, max_xyz, dataframe_cleanup=None, drop_axons=False):
    """Pick all the segments that have their midpoints inside the given voxel.

    Args:
        index (spatial_index.MultiIndex): spatial index `MultiIndex` instance
        min_xyz (np.array): 1x3 array denoting the start of voxel
        max_xyz (np.array): 1x3 array denoting the end of voxel
        dataframe_cleanup(callable (df)): function to remove any unnecessary columns and do other
            processing, *must do all operations in place*, None if not needed
        drop_axons (bool): whether to drop the axons from the queried segments

    Returns:
        pd.DataFrame: segments inside the voxel
    """
    segs_df = _sample_with_spatial_index(index, min_xyz, max_xyz)

    if drop_axons:
        # Drop axons early to save some CPU cycles
        segs_df = segs_df[segs_df["section_type"] != SectionType.axon].reset_index(drop=True)

    if len(segs_df) == 0:
        return None

    # pylint: disable=unsubscriptable-object
    starts = segs_df[SEGMENT_START_COLS].to_numpy()
    ends = segs_df[SEGMENT_END_COLS].to_numpy()
    # pylint: enable=unsubscriptable-object

    # keep only the segments whose midpoints are in the current voxel
    in_bb = pd.DataFrame((ends + starts) / 2.0, columns=XYZ)
    in_bb = in_bounding_box(*min_max_axis(min_xyz, max_xyz), df=in_bb)

    segs_df = segs_df[in_bb].copy()  # pylint: disable=unsubscriptable-object

    if len(segs_df) == 0:
        return None

    segs_df["segment_length"] = np.linalg.norm(ends[in_bb] - starts[in_bb], axis=1)
    segs_df["segment_length"] = segs_df["segment_length"].astype(np.float32)
    segs_df["section_type"] = segs_df["section_type"].astype(np.int16)

    if dataframe_cleanup is not None:
        dataframe_cleanup(segs_df)

    return segs_df


def pick_synapse_locations(segments, segment_pref, count):
    """Probabilistically pick given count of synapse locations for the given segments.

    Args:
        segments (pd.DataFrame): segments for which to pick synapses
        segment_pref (callable (df -> floats)): function to assign probabilities per segment
        count (int): desired count of synapses

    Returns:
        pd.DataFrame: the picked synapses
    """
    try:
        prob_density = normalize_probability(segment_pref(segments))
    except ErrorCloseToZero:
        return None

    picked = np.random.choice(len(segments), size=count, replace=True, p=prob_density)
    segs = segments.iloc[picked].reset_index(drop=True)
    starts = segs[SEGMENT_START_COLS].to_numpy()
    ends = segs[SEGMENT_END_COLS].to_numpy()

    alpha = np.random.random(size=len(segs)).astype(np.float32)
    segs["synapse_offset"] = alpha * segs["segment_length"]

    locations = alpha[:, np.newaxis] * starts + (1.0 - alpha)[:, np.newaxis] * ends
    locations = pd.DataFrame(locations.astype(np.float32), columns=XYZ, index=segs.index)
    return segs.join(locations)


def pick_synapses_voxel(xyz_counts, index, segment_pref, dataframe_cleanup):
    """Select `count` synapses from the circuit that lie between `min_xyz` and `max_xyz`

    Args:
        xyz_counts(tuple of min_xyz, max_xyz, count): bounding box and count of synapses desired
        index(Path): index object
        segment_pref(callable (df -> floats)): function to assign probabilities per segment
        dataframe_cleanup(callable (df)): function to remove any unnecessary columns and do other
            processing, *must do all operations in place*, None if not needed

    Returns:
        pd.DataFrame: picked synapses with columns as defined in WANTED_COLS
    """
    min_xyz, max_xyz, count = xyz_counts

    segs_df = pick_segments_voxel(index, min_xyz, max_xyz, dataframe_cleanup)

    if segs_df is None:
        return None

    segs_df = pick_synapse_locations(segs_df, segment_pref, count)

    return segs_df[WANTED_COLS] if segs_df is not None else None


def downcast_int_columns(df):
    """Downcast int columns"""
    for name in INT_COLS:
        df[name] = convert_to_smallest_allowed_int_type(df[name])


def pick_synapses_chunk(xyz_counts, index_path, segment_pref, dataframe_cleanup):
    """Pick synapses for a chunk of voxels."""
    index = spatial_index.open_index(str(index_path), max_cache_size_mb=CACHE_SIZE_MB)
    syns = []

    for it in xyz_counts:
        min_xyz, max_xyz, count = it[:3], it[3:6], int(it[6])
        syns.append(
            pick_synapses_voxel([min_xyz, max_xyz, count], index, segment_pref, dataframe_cleanup)
        )

    if all(df is None for df in syns):
        return None

    return pd.concat(syns, ignore_index=True)


def pick_synapses(
    index_path,
    xyzs_count,
    segment_pref=segment_pref_length,
    dataframe_cleanup=downcast_int_columns,
):
    """Sample segments from circuit
    Args:
        index_path(Path): absolute path spatial index `MultiIndex`
        synapse_counts(VoxelData):
            A VoxelData containing the number of segment to be sampled in each voxel

    Returns:
        pd.DataFrame: picked synapses with columns as defined in WANTED_COLS
    """
    chunks = np.array_split(xyzs_count, PARALLEL_JOBS)

    func = partial(
        pick_synapses_chunk,
        index_path=index_path,
        segment_pref=segment_pref,
        dataframe_cleanup=dataframe_cleanup,
    )

    synapses = list(map_parallelize(func, tqdm(chunks, total=len(chunks)), jobs=PARALLEL_JOBS))

    L.debug("Picking finished. Now concatenating...")
    synapses = pd.concat(synapses, ignore_index=True)

    if (percentage := 100 * len(synapses) / sum(xyzs_count[:, -1])) < 90:
        L.warning("Could only pick %.2f %% of the intended synapses", percentage)

    return synapses


def organize_indices(synapses):
    """Reorganize the synapses indices

    Change is done in place, i.e., the input dataframe is modified."""
    synapses.set_index(["tgid", "sgid"], inplace=True)
    synapses.sort_index(inplace=True)
    synapses.reset_index(inplace=True)

    return synapses
