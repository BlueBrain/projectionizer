"""Step 3: write sonata files
"""
import json
import logging
import subprocess
from pathlib import Path

import h5py
import numpy as np
from luigi import LocalTarget, Parameter

from projectionizer import analysis, write_sonata
from projectionizer.luigi_utils import (
    CommonParams,
    JsonTask,
    RunAnywayTargetTempDir,
    WriteSonata,
)
from projectionizer.step_1_assign import VirtualFibers
from projectionizer.step_2_prune import (
    ChooseConnectionsToKeep,
    ComputeAfferentSectionPos,
    ReducePrune,
)
from projectionizer.utils import XYZUVW, load, load_all

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


class CheckSonataOutput(WriteSonata):
    """Runs some sanity checks on the output files."""

    def requires(self):
        return (
            self.clone(RunParquetConverter),
            self.clone(ReducePrune),
            self.clone(WriteSonataNodes),
            self.clone(WriteSonataEdges),
        )

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
            # Might be longer if last fibers weren't picked to `syns`. Never shorter.
            assert len_nodes >= (syns.sgid.max() + 1), "Node count mismatch (feather -> h5)"

        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)


class RunAll(WriteSonata):  # pragma: no cover
    """Run all write tasks"""

    def requires(self):
        return self.clone(CheckSonataOutput), self.clone(analysis.Analyse)

    def run(self):
        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)


class WriteSonataNodes(CommonParams):
    """Write Sonata nodes file to be parameterized with Spykfunc."""

    mtype = Parameter()
    node_population = Parameter()
    node_file_name = Parameter()

    def requires(self):
        return self.clone(VirtualFibers)

    def run(self):
        fibers = load(self.input().path).reset_index()  # reset index to have `sgid` as a column
        fibers.rename(columns=dict(zip(XYZUVW, write_sonata.FIBER_COLS)), inplace=True)

        write_sonata.write_nodes(
            fibers,
            self.output().path,
            self.node_population,
            self.mtype,
        )

    def output(self):
        return LocalTarget(self.folder / self.node_file_name)


class WriteSonataEdges(CommonParams):
    """Write Sonata edges file to be parameterized with Spykfunc."""

    edge_population = Parameter()
    edge_file_name = Parameter()

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
        return LocalTarget(self.folder / ("nonparameterized-" + self.edge_file_name))


class RunSpykfunc(WriteSonata):
    """Run spykfunc for the projections."""

    def requires(self):
        return self.clone(WriteSonataEdges), self.clone(WriteSonataNodes)

    def _parse_command(self):
        spykfunc_dir = self.output().path
        edges = self.input()[0].path
        from_nodes = self.input()[1].path
        to_nodes = self.target_nodes
        cluster_dir = self.folder / "_hadoop_cluster"
        return (
            f"module purge; module load {self.module_archive} spykfunc; "
            "unset SLURM_MEM_PER_NODE; unset SLURM_MEM_PER_GPU; unset SLURM_MEM_PER_CPU; "
            f"srun dplace functionalizer --work-dir {cluster_dir} "
            f"--output-dir={spykfunc_dir} "
            f"--from {from_nodes} {self.node_population} "
            f"--to {to_nodes} {self.target_population} "
            "--filters AddID,SynapseProperties "
            f"--recipe {self.physiology_path} "
            f"--morphologies {self.morphology_path} "
            f"-- {edges} {self.edge_population} "
        )

    def run(self):
        try:
            subprocess.run(self._parse_command(), shell=True, check=True)
        except subprocess.CalledProcessError:
            if self.output().exists():
                self.output().remove()
            raise

    def output(self):
        return LocalTarget(self.folder / "spykfunc")


class RunParquetConverter(WriteSonata):
    """Run parquet converters for the spykfunc parquet files."""

    def requires(self):
        return self.clone(RunSpykfunc), self.clone(WriteSonataNodes)

    def _parse_command(self):
        parquet_dir = Path(self.input()[0].path) / "circuit.parquet"
        edge_file_name = self.output().path
        return (
            "module purge; "
            f"module load {self.module_archive} parquet-converters; "
            f"parquet2hdf5 {parquet_dir} {edge_file_name} {self.edge_population}"
        )

    def run(self):
        subprocess.run(self._parse_command(), shell=True, check=True)

    def output(self):
        return LocalTarget(self.folder / self.edge_file_name)
