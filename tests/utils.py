from pathlib import Path

import numpy as np
import pandas as pd
from bluepy import Section, Segment
from neurom import NeuriteType

from projectionizer.utils import MANIFEST_FILE

TEST_DIR = Path(__file__).parent.absolute()
TEST_DATA_DIR = TEST_DIR / "data"

NODE_POPULATION = "fake_node"
EDGE_POPULATION = "fake_edge"
CIRCUIT_CONFIG_FILE = "CircuitConfig"


def fake_segments(min_xyz, max_xyz, count):
    RADIUS = 10
    COLUMNS = [
        Segment.X1,
        Segment.Y1,
        Segment.Z1,
        Segment.X2,
        Segment.Y2,
        Segment.Z2,
        Segment.R1,
        Segment.R2,
        "gid",
        Section.ID,
        Segment.ID,
        Section.NEURITE_TYPE,
    ]

    def samp(ax):
        return (min_xyz[ax] + (max_xyz[ax] - min_xyz[ax]) * np.random.random((2, count))).T

    X, Y, Z = 0, 1, 2
    df = pd.DataFrame(index=np.arange(count), columns=COLUMNS)
    df[[Segment.X1, Segment.X2]] = samp(X)
    df[[Segment.Y1, Segment.Y2]] = samp(Y)
    df[[Segment.Z1, Segment.Z2]] = samp(Z)
    df[[Segment.R1, Segment.R2]] = (RADIUS * np.random.random((2, count))).T

    df[[Section.ID, Segment.ID, "gid"]] = np.random.randint(100, size=(3, count)).T
    df[Section.NEURITE_TYPE] = NeuriteType.apical_dendrite

    return df


def fake_manifest(path):
    manifest = "common:\n" f"  node_population_name: {NODE_POPULATION}\n"
    (path / MANIFEST_FILE).write_text(manifest)


def fake_circuit_config(path):
    config = (
        "Run Default {\n"
        "    CircuitPath fake\n"
        "    nrnPath fake\n"
        f"    MorphologyPath {path}\n"
        "    MorphologyType fake\n"
        "    METypePath fake\n"
        "    MEComboInfoFile fake\n"
        "    CellLibraryFile fake\n"
        f"    BioName {path}\n"
        "    Atlas fake\n"
        "}"
    )
    (path / CIRCUIT_CONFIG_FILE).write_text(config)
