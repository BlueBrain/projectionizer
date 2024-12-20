"""tools for writing sonata files"""

import h5py
import numpy as np
from morphio import SectionType

# Assume the source cell types to be axons
EFFERENT_SECTION_TYPE = SectionType.axon
FIBER_COLS = [
    "fiber_start_x",
    "fiber_start_y",
    "fiber_start_z",
    "fiber_direction_x",
    "fiber_direction_y",
    "fiber_direction_z",
]


def write_nodes(syns_fibers, path, population_name, mtype):
    """Write the nodes file.

    Args:
        syns_fibers (pd.DataFrame): synapses/fibers dataframe
        path (Path): nodes file path
        population_name(str): node population
        mtype (str): mtype for the nodes
    """
    sgid_count = syns_fibers.sgid.max() + 1

    with h5py.File(path, "w") as h5:
        population_path = f"/nodes/{population_name}"
        group = h5.create_group(population_path)
        group["node_type_id"] = np.full(sgid_count, -1, dtype=np.int16)
        attributes = group.create_group("0")
        attributes["mtype"] = np.full(sgid_count, 0, dtype=np.int16)

        attributes["synapse_class"] = attributes["mtype"]
        attributes["model_type"] = attributes["mtype"]
        attributes["etype"] = attributes["mtype"]
        attributes["morphology"] = attributes["mtype"]
        attributes["region"] = attributes["mtype"]

        if all(col in syns_fibers.columns for col in FIBER_COLS):
            for c in FIBER_COLS:
                attributes.create_dataset(c, data=syns_fibers[c].to_numpy(), dtype=np.float32)

        str_dt = h5py.string_dtype(encoding="utf-8")
        library = attributes.create_group("@library")
        library.create_dataset("mtype", data=[mtype], dtype=str_dt)
        library.create_dataset("synapse_class", data=["EXC"], dtype=str_dt)
        library.create_dataset("model_type", data=["virtual"], dtype=str_dt)

        library["etype"] = library["model_type"]
        library["morphology"] = library["model_type"]
        library["region"] = library["model_type"]


def write_edges(syns, path, population_name):
    """Write the edges file.

    Args:
        syns (pd.DataFrame): synapses dataframe
        path (Path): edges file path
        population_name(str): edge population
    """
    with h5py.File(path, "w") as h5:
        population_path = f"/edges/{population_name}"
        group = h5.create_group(population_path)

        # source_node_id indexing starts at 0
        group["source_node_id"] = syns.sgid.to_numpy()
        group["target_node_id"] = syns.tgid.to_numpy()
        group["edge_type_id"] = np.full(len(syns), -1, dtype=np.int16)

        attributes = group.create_group("0")
        attributes["afferent_section_type"] = syns.section_type.to_numpy()
        attributes["efferent_section_type"] = np.full(
            len(syns), EFFERENT_SECTION_TYPE, dtype=np.int16
        )

        attributes["distance_soma"] = syns.sgid_path_distance.to_numpy()
        attributes["afferent_section_id"] = syns.section_id.to_numpy()
        attributes["afferent_section_pos"] = syns.section_pos.to_numpy()
        attributes["afferent_segment_id"] = syns.segment_id.to_numpy()
        attributes["afferent_segment_offset"] = syns.synapse_offset.to_numpy()

        attributes["afferent_center_x"] = syns.x.to_numpy()
        attributes["afferent_center_y"] = syns.y.to_numpy()
        attributes["afferent_center_z"] = syns.z.to_numpy()

        if all(np.in1d(["source_x", "source_y", "source_z"], syns.columns)):
            attributes["efferent_center_x"] = syns.source_x.to_numpy()
            attributes["efferent_center_y"] = syns.source_y.to_numpy()
            attributes["efferent_center_z"] = syns.source_z.to_numpy()

        if "distance_volume_transmission" in syns.columns:
            attributes["distance_volume_transmission"] = syns.distance_volume_transmission
