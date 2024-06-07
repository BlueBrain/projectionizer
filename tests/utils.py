import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from morphio import SectionType

TEST_DIR = Path(__file__).parent.absolute()
TEST_DATA_DIR = TEST_DIR / "data"

NODE_POPULATION = "fake_node"
EDGE_POPULATION = "fake_edge"
CIRCUIT_CONFIG_FILE = "CircuitConfig"


def as_iterable(item):
    if not item:
        return []
    return item if isinstance(item, Iterable) else [item]


def fake_segments(min_xyz, max_xyz, count):
    COLUMNS = [
        "segment_x1",
        "segment_y1",
        "segment_z1",
        "segment_x2",
        "segment_y2",
        "segment_z2",
        "gid",
        "section_id",
        "segment_id",
        "section_type",
    ]

    def samp(ax):
        return (min_xyz[ax] + (max_xyz[ax] - min_xyz[ax]) * np.random.random((2, count))).T

    X, Y, Z = 0, 1, 2
    df = pd.DataFrame(index=np.arange(count), columns=COLUMNS)
    df[["segment_x1", "segment_x2"]] = samp(X)
    df[["segment_y1", "segment_y2"]] = samp(Y)
    df[["segment_z1", "segment_z2"]] = samp(Z)

    df[["section_id", "segment_id", "gid"]] = np.random.randint(100, size=(3, count)).T
    df["section_type"] = SectionType.apical_dendrite

    return df.copy()


def fake_circuit_config(path):
    config = {
        "components": {"alternate_morphologies": {"neurolucida-asc": str(TEST_DATA_DIR)}},
        "networks": {
            "nodes": [{"nodes_file": "fake_path.h5", "populations": {"fake": {}}}],
            "edges": [],
        },
        "metadata": {"status": "partial"},
    }
    (path / CIRCUIT_CONFIG_FILE).write_text(json.dumps(config))
