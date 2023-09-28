"""SSCX functions related to working w/ hex"""
import numpy as np
import pandas as pd

from projectionizer.utils import XYZUVW


def get_virtual_fiber_locations(bounding_box, locations_path):
    """get locations in bounding box of central column"""

    if not len(bounding_box):
        return []

    min_xz, max_xz = bounding_box

    locations = pd.read_csv(locations_path)[["x", "z"]].to_numpy()

    idx = np.all((min_xz <= locations) & (locations <= max_xz), axis=1)
    return locations[idx]


def get_fibers_in_region(mask, height, locations_path):
    """Get fibers that are located inside the region (mask)"""
    fibers = pd.read_csv(locations_path)[list("xyz")]

    mask_xz = mask.any(axis=1)
    ind = height.positions_to_indices(fibers.to_numpy())
    ret = fibers[mask_xz[tuple(ind[:, [0, 2]].T)]]
    return ret[list("xz")].to_numpy()


def get_minicol_virtual_fibers(apron_bounding_box, height, region_mask, locations_path):
    """Get fibers for a column.

    Returns:
        pd.Dataframe: Fibers in a dataframe. Columns are position and direction vectors of the
            fibers as well as a boolean denoting whether or not the fiber lays on the apron.
    """
    fibers = set(tuple(loc) for loc in get_fibers_in_region(region_mask, height, locations_path))
    extra_fibers = set(
        tuple(loc)
        for loc in get_virtual_fiber_locations(
            bounding_box=apron_bounding_box, locations_path=locations_path
        )
    )
    extra_fibers = extra_fibers - fibers

    def to_dataframe(points, is_apron):
        """return fibers in a dataframe"""
        df = pd.DataFrame(columns=XYZUVW, dtype=float)
        if not points:
            return df
        df.x, df.z = zip(*points)
        df.v = 1.0  # all direction vectors point straight up
        df["apron"] = is_apron
        df["apron"] = df["apron"].astype(bool)
        return df.fillna(0)

    return pd.concat(
        (to_dataframe(fibers, False), to_dataframe(extra_fibers, True)),
        ignore_index=True,
        sort=True,
    )


def get_mask_bounding_box(distance, mask, bounding_box):
    """return a mask of the area/volume covered by the bounding box"""

    mask_bb = np.full_like(mask, False, dtype=bool)

    # Add y for the bounding box
    bb = np.copy(bounding_box)
    min_xyz = np.insert(bb[0], 1, distance.offset[1])
    max_xyz = np.insert(bb[1], 1, distance.offset[1])

    # Get x and z indexes for bounding box
    min_ind = distance.positions_to_indices(min_xyz)
    max_ind = distance.positions_to_indices(max_xyz) + 1

    # Get y index for bounding box
    min_ind[1] = np.where(mask)[1].min()
    max_ind[1] = np.where(mask)[1].max() + 1

    mask_bb[min_ind[0] : max_ind[0], min_ind[1] : max_ind[1], min_ind[2] : max_ind[2]] = True

    return mask_bb
