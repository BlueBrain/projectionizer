"""tools for writing sonata files"""
import h5py
import numpy as np
from six import text_type


def write_nodes(syns, path, population_name, mtype, keep_offset=True):
    """write the nodes file"""
    if keep_offset:
        sgid_count = syns.sgid.max()
    else:
        sgid_count = syns.sgid.max() - syns.sgid.min() + 1

    with h5py.File(path, "w") as h5:
        population_path = f"/nodes/{population_name}"
        group = h5.create_group(population_path)
        group["node_type_id"] = np.full((sgid_count,), -1, dtype=np.int8)
        attributes = group.create_group("0")
        attributes["mtype"] = np.full((sgid_count,), 0, dtype=np.int8)

        attributes["synapse_class"] = attributes["mtype"]
        attributes["model_type"] = attributes["mtype"]
        attributes["etype"] = attributes["mtype"]
        attributes["morphology"] = attributes["mtype"]
        attributes["region"] = attributes["mtype"]

        library = attributes.create_group("@library")
        str_dt = h5py.special_dtype(vlen=text_type)
        library.create_dataset(
            "mtype",
            data=[
                mtype,
            ],
            dtype=str_dt,
        )
        library.create_dataset(
            "synapse_class",
            data=[
                "EXC",
            ],
            dtype=str_dt,
        )
        library.create_dataset(
            "model_type",
            data=[
                "virtual",
            ],
            dtype=str_dt,
        )

        library["etype"] = library["model_type"]
        library["morphology"] = library["model_type"]
        library["region"] = library["model_type"]


def write_edges(syns, path, population_name, keep_offset=True):
    """write the edges file"""
    if keep_offset:
        min_sgid = 1
    else:
        min_sgid = syns.sgid.min()

    with h5py.File(path, "w") as h5:
        population_path = f"/edges/{population_name}"
        group = h5.create_group(population_path)

        group["source_node_id"] = syns.sgid.to_numpy() - min_sgid
        group["target_node_id"] = syns.tgid.to_numpy() - 1
        group["edge_type_id"] = np.full((len(syns),), -1, dtype=np.int8)

        attributes = group.create_group("0")
        attributes["distance_soma"] = syns.sgid_path_distance.to_numpy()

        attributes["afferent_section_id"] = syns.section_id.to_numpy()
        attributes["afferent_segment_id"] = syns.segment_id.to_numpy()
        attributes["afferent_segment_offset"] = syns.synapse_offset.to_numpy()

        attributes["afferent_center_x"] = syns.x.to_numpy()
        attributes["afferent_center_y"] = syns.y.to_numpy()
        attributes["afferent_center_z"] = syns.z.to_numpy()
