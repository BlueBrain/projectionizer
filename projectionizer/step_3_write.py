"""Step 3: write sonata files
"""
import json
import logging
import os
import re
import shutil
import subprocess

import h5py
import numpy as np
from luigi import Parameter
from luigi.local_target import LocalTarget

from projectionizer import write_sonata
from projectionizer.luigi_utils import CommonParams, JsonTask, RunAnywayTargetTempDir
from projectionizer.step_2_prune import (
    ChooseConnectionsToKeep,
    ComputeAfferentSectionPos,
    ReducePrune,
)
from projectionizer.utils import load, load_all

L = logging.getLogger(__name__)


class SynapseCountPerConnectionTarget(JsonTask):  # pragma: no cover
    """Compute the mean number of synapses per connection for L4 PC cells"""

    def requires(self):
        return self.clone(ChooseConnectionsToKeep)

    def run(self):
        connections = load(self.input().path)

        mask = connections.mtype.isin(self.target_mtypes) & connections.kept
        mean = connections[mask].connection_size.mean()
        if np.isnan(mean):
            raise ValueError("SynapseCountPerConnectionTarget returned NaN")
        with self.output().open("w") as outputf:
            json.dump({"result": mean}, outputf)


class WriteSonata(CommonParams):
    """Write projections in SONATA format."""

    mtype = Parameter("projections")
    node_population = Parameter("projections")
    edge_population = Parameter("projections")
    node_file_name = Parameter("projections-nodes.h5")
    edge_file_name = Parameter("projections-edges.h5")

    def requires(self):
        return (
            self.clone(RunParquetConverter),
            self.clone(ReducePrune),
            self.clone(WriteSonataNodes),
            self.clone(WriteSonataEdges),
        )

    def _get_full_path_output(self, filename):
        """Get full path (required by spykfunc) for output files."""
        return os.path.realpath(os.path.join(self.folder, filename))

    def run(self):
        """Run checks on resulting files to eradicate e.g., off-by-1 errors."""
        edge_file = self.input()[0].path
        syns = load(self.input()[1].path)
        node_file = self.input()[2].path
        nonparameterized_edge_file = self.input()[3].path

        with h5py.File(edge_file, "r") as h5:
            population = h5[f"edges/{self.edge_population}"]
            len_edges = len(population["source_node_id"])
        with h5py.File(nonparameterized_edge_file, "r") as h5:
            population = h5[f"edges/{self.edge_population}"]
            len_np_edges = len(population["source_node_id"])
            assert len_np_edges == len_edges, "Edge count mismatch (parameterized)"

            assert np.all(population["source_node_id"][:] == syns.sgid), (
                "SGID conversion mismatch."
                "Unnecessary 1 to 0-based index conversion? (feather -> h5)"
            )

            assert np.all(
                population["target_node_id"][:] == syns.tgid
            ), "TGID mismatch. Unnecessary 1 to 0-based index conversion? (feather -> h5)"

        with h5py.File(node_file, "r") as h5:
            population = h5[f"nodes/{self.node_population}"]
            len_nodes = len(population["node_type_id"])
            assert len_nodes == (syns.sgid.max() + 1), "Node count mismatch (feather -> h5)"

    def output(self):
        return LocalTarget(self.input()[0].path)


class WriteAll(CommonParams):  # pragma: no cover
    """Run all write tasks"""

    def requires(self):
        return self.clone(WriteSonata)

    def run(self):
        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)


class WriteSonataNodes(WriteSonata):
    """Write Sonata nodes file to be parameterized with Spykfunc."""

    def requires(self):
        return self.clone(ReducePrune)

    def run(self):
        write_sonata.write_nodes(
            load(self.input().path),
            self.output().path,
            self.node_population,
            self.mtype,
        )

    def output(self):
        return LocalTarget(self._get_full_path_output(self.node_file_name))


class WriteSonataEdges(WriteSonata):
    """Write Sonata edges file to be parameterized with Spykfunc."""

    def requires(self):
        return self.clone(ReducePrune), self.clone(ComputeAfferentSectionPos)

    def run(self):
        synapses, section_pos = load_all(self.input())
        write_sonata.write_edges(
            synapses.join(section_pos),
            self.output().path,
            self.edge_population,
        )

    def output(self):
        return LocalTarget(self._get_full_path_output("nonparameterized-" + self.edge_file_name))


def _check_if_old_syntax(archive):
    """Check if old command format needs to be used with spykfunc and parquet-converters.

    New format is expected starting from archive/2021-07."""
    m = re.match(r"archive/(?P<year>\d+)-(?P<month>\d+)", archive)
    if m is None:
        return False
    year = m.group("year")
    month = m.group("month")
    return year < "2021" or (year == "2021" and month < "07")


class RunSpykfunc(WriteSonata):
    """Run spykfunc for the projections."""

    def requires(self):
        return self.clone(WriteSonataEdges), self.clone(WriteSonataNodes)

    def _parse_command(self):
        spykfunc_dir = self.output().path
        edges = self.input()[0].path
        from_nodes = self.input()[1].path
        to_nodes = self.target_nodes
        cluster_dir = self._get_full_path_output("_sm_cluster")
        command = (
            f"module purge; module load {self.module_archive} spykfunc; "
            f"unset SLURM_MEM_PER_NODE; unset SLURM_MEM_PER_GPU; unset SLURM_MEM_PER_CPU; "
            f"sm_run -m 0 -w {cluster_dir} spykfunc --output-dir={spykfunc_dir} "
            f"-p spark.master=spark://$(hostname):7077 "
            f"--from {from_nodes} {self.node_population} "
            f"--to {to_nodes} {self.target_population} "
            f"--filters AddID,SynapseProperties "
        )

        if _check_if_old_syntax(self.module_archive):
            command += (
                f"--touches {edges} {self.edge_population} "
                f"{self.physiology_path} "
                f"{self.morphology_path} "
            )
        else:
            command += (
                f"'--recipe' {self.physiology_path} "
                f"'--morphologies' {self.morphology_path} "
                f"{edges} {self.edge_population} "
            )

        return command

    def run(self):
        try:
            subprocess.run(self._parse_command(), shell=True, check=True)
        except subprocess.CalledProcessError:
            if os.path.isdir(self.output().path):
                shutil.rmtree(self.output().path)
            raise

    def output(self):
        return LocalTarget(self._get_full_path_output("spykfunc"))


class RunParquetConverter(WriteSonata):
    """Run parquet converters for the spykfunc parquet files."""

    def requires(self):
        return self.clone(RunSpykfunc), self.clone(WriteSonataNodes)

    def _parse_command(self):
        from_nodes = self.input()[1].path
        parquet_dir = os.path.join(self.input()[0].path, "circuit.parquet")
        parquet_glob = os.path.join(parquet_dir, "*.parquet")
        to_nodes = self.target_nodes
        edge_file_name = self.output().path
        command = (
            f"module purge; "
            f"module load {self.module_archive} parquet-converters; "
            f"parquet2hdf5 "
        )

        if _check_if_old_syntax(self.module_archive):
            command += (
                f"--format SONATA "
                f"--from {from_nodes} {self.node_population} "
                f"--to {to_nodes} {self.target_population} "
                f"'-o' {edge_file_name} "
                f"'-p' {self.edge_population} "
                f"{parquet_glob} "
            )
        else:
            command += f"{parquet_dir} {edge_file_name} {self.edge_population}"

        return command

    def run(self):
        subprocess.run(self._parse_command(), shell=True, check=True)

    def output(self):
        return LocalTarget(self._get_full_path_output(self.edge_file_name))
