"""Luigi related utils"""

import re
from pathlib import Path

import importlib_resources
from bluepysnap import Circuit
from luigi import (
    Config,
    FloatParameter,
    IntParameter,
    ListParameter,
    Parameter,
    PathParameter,
    Task,
)
from luigi.contrib.simulate import RunAnywayTarget
from luigi.local_target import LocalTarget

from projectionizer.version import VERSION

REGEX_VERSION = re.compile(r"^\d+\.\d+\.\d+")
TEMPLATES_PATH = importlib_resources.files(__package__) / "templates"
MINIMUM_ARCHIVE = "archive/2023-06"
MORPH_TYPES = {"h5", "asc", "swc"}


def _check_module_archive(archive):
    m = re.match(r"^archive/\d{4}-\d{2}$", archive)
    if archive != "unstable" and (m is None or archive < MINIMUM_ARCHIVE):
        raise ValueError(
            f"Invalid module archive: '{archive}'. "
            f"Expected 'unstable' or 'archive/YYYY-MM' >= '{MINIMUM_ARCHIVE}'"
        )


def _check_version_compatibility(version):
    if match := REGEX_VERSION.match(version):
        curr_version = REGEX_VERSION.match(VERSION).group()
        if match.group() == curr_version:
            return
        raise RuntimeError(
            f"Given config file is intended for projectionizer version '{version}'. "
            f"However, the version of the running projectionizer is '{VERSION}'.\n\n"
            "Continueing runs with mixed versions of projectionizer is strongly discouraged.\n\n"
            "To update the config to match the version, please see:\n\n"
            f"https://bbpteam.epfl.ch/documentation/projects/projectionizer/{curr_version}/"
        )

    raise ValueError(
        "Expected projectionizer version to be given in format 'X.Y.Z' or 'X.Y.Z.devN', "
        f"got: '{version}'."
    )


def camel2spinal_case(name):
    """Camel case to snake case"""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()


class FolderTask(Task):
    """Simple dependency task to create missing folders"""

    folder = PathParameter(absolute=True)

    def run(self):
        self.folder.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member

    def output(self):
        return LocalTarget(self.folder)


class CommonParams(Config):
    """Parameters that must be passed to all Tasks"""

    atlas_path = PathParameter(absolute=True, exists=True)
    projectionizer_version = Parameter()
    circuit_config = PathParameter(absolute=True, exists=True)
    physiology_path = PathParameter(absolute=True, exists=True)
    morphology_type = Parameter(default="asc")
    segment_index_path = PathParameter(absolute=True, exists=True)
    folder = PathParameter(absolute=True)
    n_total_chunks = IntParameter()
    oversampling = FloatParameter()
    layers = ListParameter()  # list of pairs of (layer name, thickness), starting at 'bottom'
    target_population = Parameter()
    target_mtypes = ListParameter(
        default=[
            "L4_PC",
            "L4_UPC",
            "L4_TPC",
        ]
    )  # list of mtypes
    regions = ListParameter()

    # path to CSV with six columns; x,y,z,u,v,w: location and direction of fibers
    fiber_locations_path = PathParameter(
        absolute=True,
        exists=True,
        default=TEMPLATES_PATH / "rat_fibers.csv",
    )

    # module archive from which to load spykfunc, parquet-converters
    module_archive = Parameter(default=MINIMUM_ARCHIVE)

    # hex parameters
    # bounding box for apron around the hexagon, so that there aren't edge effects when assigning
    # synapses to fibers
    # ListParameter can not default to None without further problems with luigi
    hex_apron_bounding_box = ListParameter(default=[])

    extension = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        node_population = Circuit(self.circuit_config).nodes[self.target_population]

        if self.morphology_type not in MORPH_TYPES:
            raise ValueError(
                f"morphology_type '{self.morphology_type}' is not "
                f"one of {' / '.join(sorted(MORPH_TYPES))}"
            )

        _check_module_archive(self.module_archive)

        self.morphology_path = Path(node_population.morph.get_morphology_dir(self.morphology_type))
        self.target_nodes = Path(node_population.h5_filepath)

        _check_version_compatibility(self.projectionizer_version)

    def output(self):
        name = camel2spinal_case(self.__class__.__name__)
        target = f"{self.folder}/{name}.{self.extension}"
        if hasattr(self, "chunk_num"):
            target = f"{self.folder}/{name}-{getattr(self, 'chunk_num')}.{self.extension}"
        return LocalTarget(target)

    def requires(self):
        return FolderTask(folder=self.folder)

    @staticmethod
    def load_data(path):
        """completely unqualified paths are loaded from the templates directory"""
        if "/" in path:
            return path
        else:
            return TEMPLATES_PATH / path


class CsvTask(CommonParams):
    """Task returning a CSV file"""

    extension = "csv"


class FeatherTask(CommonParams):
    """Task returning a feather file"""

    extension = "feather"


class JsonTask(CommonParams):
    """Task returning a JSON file"""

    extension = "json"


class NrrdTask(CommonParams):
    """Task returning a Nrrd file"""

    extension = "nrrd"


class RunAnywayTargetTempDir(RunAnywayTarget):
    """Override tmp directory location for RunAnywayTarget

    RunAnywayTarget uses a directory in /tmp for keeping state,
    so if two different users try and launch a task that uses
    this target, it fails.  By using this target, the directory
    is under the user's control, and thus there won't be conflicts
    """

    def __init__(self, task_obj, base_dir):
        self.temp_dir = base_dir / "luigi-tmp"
        super().__init__(task_obj)


# NOTE: Keeping the name WriteSonata to have smaller impact on the old configs.
class WriteSonata(CommonParams):
    """A common place of inheritance for different projectionizer tasks."""

    mtype = Parameter("projections")
    node_population = Parameter("projections")
    edge_population = Parameter("projections")
    node_file_name = Parameter("projections-nodes.h5")
    edge_file_name = Parameter("projections-edges.h5")
