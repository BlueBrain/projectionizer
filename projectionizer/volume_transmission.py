"""Volume Transmission luigi tasks"""

import logging
import shutil
from functools import partial

import h5py
import numpy as np
import pandas as pd
import spatial_index.experimental
from luigi import FloatParameter, ListParameter, LocalTarget
from tqdm import tqdm

from projectionizer import step_1_assign, step_2_prune, step_3_write
from projectionizer.luigi_utils import FeatherTask, WriteSonata
from projectionizer.straight_fibers import calc_pathlength_to_fiber_start
from projectionizer.synapses import (
    CACHE_SIZE_MB,
    PARALLEL_JOBS,
    spatial_index,
    spherical_sampling,
)
from projectionizer.utils import (
    XYZ,
    XYZUVW,
    calculate_conductance_scaling_factor,
    delete_file_on_exception,
    load,
    map_parallelize,
    write_feather,
)

L = logging.getLogger(__name__)

MAX_VOLUME_TRANSMISSION_DISTANCE = 5
DEFAULT_ADDITIVE_PATH_DISTANCE = 300
EDGE_FILE_NAME = "volume-transmission-edges.h5"


def _sample_chunk_spherical(chunk, index_path, radius):
    """Perform spherical sampling on a chunk of positions."""
    index = spatial_index.open_index(str(index_path), max_cache_size_mb=CACHE_SIZE_MB)
    syns = [spherical_sampling((np.array(pos), int(sgid)), index, radius) for *pos, sgid in chunk]

    return pd.concat(syns, ignore_index=True) if syns else None


def _get_spherical_samples(syns, index_path, radius):
    """Helper function for the spherical sampling."""
    L.info("Starting spherical sampling with a radius of %s um...", radius)

    func = partial(_sample_chunk_spherical, index_path=index_path, radius=radius)
    pos = syns[list("xyz")].to_numpy()
    sgid = syns["sgid"].to_numpy()
    pos_sgid = np.hstack((pos, sgid[:, np.newaxis]))

    order = spatial_index.experimental.space_filling_order(pos)
    chunks = np.array_split(pos_sgid[order], PARALLEL_JOBS)

    samples = map_parallelize(func, tqdm(chunks))

    L.info("Concatenating samples...")
    return pd.concat(samples, ignore_index=True)


class VolumeSample(FeatherTask):
    """Spherical sampling for Volume Transmission projections."""

    radius = FloatParameter(MAX_VOLUME_TRANSMISSION_DISTANCE)
    # NOTE:
    # Maybe should be combined with PruneChunk['additive_path_distance']
    additive_path_distance = FloatParameter(DEFAULT_ADDITIVE_PATH_DISTANCE)

    def requires(self):  # pragma: no cover
        return self.clone(step_2_prune.ReducePrune), self.clone(step_1_assign.VirtualFibers)

    def run(self):
        samples = _get_spherical_samples(
            load(self.input()[0].path), self.segment_index_path, self.radius
        )
        samples.rename(columns={"gid": "tgid"}, inplace=True)

        fibers = load(self.input()[1].path)
        distances = calc_pathlength_to_fiber_start(
            samples[XYZ].to_numpy(), fibers.loc[samples.sgid][XYZUVW].to_numpy()
        )
        samples["sgid_path_distance"] = distances + self.additive_path_distance

        L.info("Writing %s...", self.output().path)
        write_feather(self.output().path, samples)


class VolumeComputeAfferentSectionPos(step_2_prune.ComputeAfferentSectionPos):
    """Computes the afferent section position for the Volume Transmission synapses"""

    def requires(self):  # pragma: no cover
        return self.clone(VolumeSample)


class ScaleConductance(WriteSonata):
    """Scale the conductance."""

    interval = ListParameter([1.0, 0.1])

    def requires(self):  # pragma: no cover
        return (self.clone(VolumeRunParquetConverter), self.clone(VolumeSample))

    def run(self):
        L.info("Scaling conductance according to distance...")
        edge_population = self.requires()[0].edge_population
        radius = self.requires()[1].radius
        filepath = self.output().path

        with delete_file_on_exception(filepath):
            shutil.copyfile(self.input()[0].path, filepath)

            with h5py.File(filepath, "r+") as projections:
                conductance = projections[f"edges/{edge_population}/0/conductance"]
                distances = np.asarray(
                    projections[f"edges/{edge_population}/0/distance_volume_transmission"]
                )
                conductance[...] = calculate_conductance_scaling_factor(
                    distances, radius, self.interval
                )

    def output(self):
        return LocalTarget(f"{self.folder}/{EDGE_FILE_NAME}")


class VolumeWriteSonataEdges(step_3_write.WriteSonataEdges):  # pragma: no cover
    """Adapter class to step_3_write.WriteSonataEdges"""

    def requires(self):
        return self.clone(VolumeSample), self.clone(VolumeComputeAfferentSectionPos)


class VolumeCheckSonataOutput(step_3_write.CheckSonataOutput):  # pragma: no cover
    """Adapter class to step_3_write.CheckSonataOutput"""

    edge_file_name = EDGE_FILE_NAME
    edge_population = "volume_projections"

    def requires(self):
        return (
            self.clone(ScaleConductance),
            self.clone(VolumeSample),
            self.clone(step_3_write.WriteSonataNodes),
            self.clone(VolumeWriteSonataEdges),
        )


class VolumeRunAll(step_3_write.RunAll):  # pragma: no cover
    """Adapter class to step_3_write.RunAll"""

    def requires(self):
        # Also run the main workflow (step_3_write.RunAll)
        return self.clone(VolumeCheckSonataOutput), self.clone(step_3_write.RunAll)


class VolumeRunSpykfunc(step_3_write.RunSpykfunc):  # pragma: no cover
    """Adapter class to step_3_write.RunSpykfunc"""

    def requires(self):
        return self.clone(VolumeWriteSonataEdges), self.clone(step_3_write.WriteSonataNodes)

    def output(self):
        return LocalTarget(self.folder / "volume-spykfunc")


class VolumeRunParquetConverter(step_3_write.RunParquetConverter):  # pragma: no cover
    """Adapter class to step_3_write.RunParquetConverter"""

    edge_file_name = "nonscaled-" + EDGE_FILE_NAME

    def requires(self):
        return self.clone(VolumeRunSpykfunc), self.clone(step_3_write.WriteSonataNodes)
