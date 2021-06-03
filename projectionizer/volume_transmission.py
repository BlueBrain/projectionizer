"""Volume Transmission luigi tasks"""
from functools import partial
import logging
import os
import shutil

from bluepy import Section, Segment
import h5py
from luigi import FloatParameter, ListParameter, LocalTarget
import pandas as pd
from tqdm import tqdm

from projectionizer.luigi_utils import CommonParams, FeatherTask
from projectionizer import analysis, step_2_prune, step_3_write
from projectionizer.straight_fibers import calc_pathlength_to_fiber_start
from projectionizer.synapses import spherical_sampling
from projectionizer.utils import (calculate_synapse_conductance,
                                  load,
                                  map_parallelize,
                                  write_feather,
                                  XYZ,
                                  XYZUVW)

L = logging.getLogger(__name__)

MAX_VOLUME_TRANSMISSION_DISTANCE = 5
DEFAULT_ADDITIVE_PATH_DISTANCE = 300


def _get_spherical_samples(syns, circuit_path, radius):
    """Helper function for the spherical sampling."""
    L.info('Starting spherical sampling with a radius of %s um...', radius)
    func = partial(spherical_sampling, index_path=circuit_path, radius=radius)
    pos = syns[list('xyz')].to_numpy()
    sgid = syns['sgid'].to_numpy()
    samples = map_parallelize(func, tqdm(zip(pos, sgid), total=len(pos)))

    L.info('Concatenating samples...')
    return pd.concat(samples, ignore_index=True)


class MainSonataWorkflow(CommonParams):  # pragma: no cover
    """Task to run the tasks regarding "normal" SONATA projections."""

    def requires(self):
        return (self.clone(step_3_write.WriteSonata),
                self.clone(step_3_write.WriteUserTargetTxt),
                self.clone(analysis.Analyse))

    def output(self):
        return LocalTarget(self.input()[0].path)


class VolumeSample(FeatherTask):
    """Spherical sampling for Volume Transmission projections."""
    radius = FloatParameter(MAX_VOLUME_TRANSMISSION_DISTANCE)
    # NOTE by herttuai on 26/08/2021:
    # Maybe should be combined with PruneChunk['additive_path_distance']
    additive_path_distance = FloatParameter(DEFAULT_ADDITIVE_PATH_DISTANCE)

    def requires(self):  # pragma: no cover
        return self.clone(step_2_prune.ReducePrune), self.clone(step_3_write.VirtualFibers)

    def run(self):
        samples = _get_spherical_samples(load(self.input()[0].path),
                                         os.path.dirname(self.circuit_config),
                                         self.radius)
        samples.rename(columns={'gid': 'tgid',
                                Section.ID: 'section_id',
                                Segment.ID: 'segment_id'}, inplace=True)

        fibers = load(self.input()[1].path)
        distances = calc_pathlength_to_fiber_start(samples[XYZ].to_numpy(),
                                                   fibers.loc[samples.sgid][XYZUVW].to_numpy())
        samples['sgid_path_distance'] = distances + self.additive_path_distance

        L.info('Writing %s...', self.output().path)
        write_feather(self.output().path, samples)


class ScaleConductance(CommonParams):
    """Scale the conductance."""
    interval = ListParameter([1.0, 0.1])

    def requires(self):  # pragma: no cover
        return self.clone(VolumeWriteSonata), self.clone(VolumeSample)

    def run(self):
        L.info('Scaling conductance according to distance...')
        syns = load(self.input()[1].path)
        edge_population = self.requires()[0].edge_population
        radius = self.requires()[1].radius

        try:
            filepath = self.output().path
            shutil.copyfile(self.input()[0].path, filepath)

            with h5py.File(filepath, 'r+') as projections:
                conductance = projections[f'edges/{edge_population}/0/conductance']
                conductance[...] = calculate_synapse_conductance(
                    conductance[:],
                    syns.distance_volume_transmission,
                    radius,
                    self.interval)
        except Exception:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise

    def output(self):
        name, ext = os.path.splitext(self.input()[0].path)
        return LocalTarget(name + '-scaled' + ext)


class VolumeWriteSonataEdges(step_3_write.WriteSonataEdges):  # pragma: no cover
    """Adapter class to step_3_write.WriteSonataEdges"""

    def requires(self):
        return self.clone(VolumeSample)


class VolumeWriteSonataNodes(step_3_write.WriteSonataNodes):  # pragma: no cover
    """Adapter class to step_3_write.WriteSonataNodes"""

    def requires(self):
        return self.clone(VolumeSample)


class VolumeWriteSonata(step_3_write.WriteSonata):  # pragma: no cover
    """Adapter class to step_3_write.WriteSonata"""
    node_file_name = 'volume-transmission-nodes.h5'
    edge_file_name = 'volume-transmission-edges.h5'
    mtype = 'volume_projections'
    node_population = 'volume_projections'
    edge_population = 'volume_projections'

    def requires(self):
        return (self.clone(VolumeRunParquetConverter),
                self.clone(VolumeSample),
                self.clone(VolumeWriteSonataNodes),
                self.clone(VolumeWriteSonataEdges))


class VolumeWriteAll(step_3_write.WriteAll):  # pragma: no cover
    """Adapter class to step_3_write.WriteAll"""

    def requires(self):
        return self.clone(ScaleConductance), self.clone(MainSonataWorkflow)


class VolumeRunSpykfunc(step_3_write.RunSpykfunc):  # pragma: no cover
    """Adapter class to step_3_write.RunSpykfunc"""

    def requires(self):
        return self.clone(VolumeWriteSonataEdges), self.clone(VolumeWriteSonataNodes)

    def output(self):
        return LocalTarget(self._get_full_path_output('volume-spykfunc'))


class VolumeRunParquetConverter(step_3_write.RunParquetConverter):  # pragma: no cover
    """Adapter class to step_3_write.RunParquetConverter"""

    def requires(self):
        return self.clone(VolumeRunSpykfunc), self.clone(VolumeWriteSonataNodes)
