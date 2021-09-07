'''Step 3: write sonata files
'''
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
from projectionizer.luigi_utils import CommonParams, CsvTask, JsonTask, RunAnywayTargetTempDir
from projectionizer.step_1_assign import VirtualFibersNoOffset
from projectionizer.step_2_prune import ChooseConnectionsToKeep, ReducePrune
from projectionizer import write_sonata
from projectionizer.utils import load

L = logging.getLogger(__name__)


def write_user_target(output, synapses, name):
    '''write target file

    Args:
        output(path): path of file to create
        synapses(dataframe): synapses
        name(str): name of target
    '''
    with open(output, 'w', encoding='utf-8') as fd:
        fd.write('Target Cell %s {\n' % name)
        for tgid in sorted(synapses.sgid.unique()):
            fd.write('    a{}\n'.format(tgid))
        fd.write('}\n')


class VirtualFibers(CsvTask):
    '''Same as VirtualFibersNoOffset but with the sgid_offset'''

    def requires(self):  # pragma: no cover
        return self.clone(VirtualFibersNoOffset)

    def run(self):  # pragma: no cover
        fibers = load(self.input().path)
        fibers.index += self.sgid_offset  # pylint: disable=maybe-no-member
        # Saving as csv because feather does not support index offset
        fibers.to_csv(self.output().path, index_label='sgid')


class SynapseCountPerConnectionTarget(JsonTask):  # pragma: no cover
    '''Compute the mean number of synapses per connection for L4 PC cells'''

    def requires(self):
        return self.clone(ChooseConnectionsToKeep)

    def run(self):
        connections = load(self.input().path)

        mask = connections.mtype.isin(self.target_mtypes) & connections.kept
        mean = connections[mask].connection_size.mean()
        if np.isnan(mean):
            raise Exception('SynapseCountPerConnectionTarget returned NaN')
        with self.output().open('w') as outputf:
            json.dump({'result': mean}, outputf)


class WriteSonata(CommonParams):
    """Write projections in SONATA format."""
    target_population = Parameter('All')
    mtype = Parameter('projections')
    node_population = Parameter('projections')
    edge_population = Parameter('projections')
    node_file_name = Parameter('projections_nodes.h5')
    edge_file_name = Parameter('projections_edges.h5')
    module_archive = Parameter('archive/2021-07')

    def requires(self):
        return (self.clone(RunParquetConverter),
                self.clone(ReducePrune),
                self.clone(WriteSonataNodes),
                self.clone(WriteSonataEdges),
                self.clone(WriteUserTargetTxt))

    def _get_full_path_output(self, filename):
        """Get full path (required by spykfunc) for output files."""
        return os.path.realpath(os.path.join(self.folder, filename))

    def run(self):
        """Run checks on resulting files to eradicate e.g., off-by-1 errors."""
        edge_file = self.input()[0].path
        syns = load(self.input()[1].path)
        node_file = self.input()[2].path
        nonparameterized_edge_file = self.input()[3].path

        with h5py.File(edge_file, 'r') as h5:
            population = h5[f'edges/{self.edge_population}']
            len_edges = len(population['source_node_id'])
        with h5py.File(nonparameterized_edge_file, 'r') as h5:
            population = h5[f'edges/{self.edge_population}']
            len_np_edges = len(population['source_node_id'])
            assert len_np_edges == len_edges, 'Edge count mismatch (parameterized)'

            assert np.all(population['source_node_id'][:] == (syns.sgid - 1)), \
                'SGID conversion mismatch (feather -> h5)'

            assert np.all(population['target_node_id'][:] == (syns.tgid - 1)), \
                'TGID conversion mismatch (feather -> h5)'

        with h5py.File(node_file, 'r') as h5:
            population = h5[f'nodes/{self.node_population}']
            len_nodes = len(population['node_type_id'])
            assert len_nodes == syns.sgid.max(), 'Node count mismatch (feather -> h5)'

    def output(self):
        return LocalTarget(self.input()[0].path)


class WriteUserTargetTxt(WriteSonata):
    '''write user.target'''

    def requires(self):
        return self.clone(ReducePrune)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        synapses = load(self.input().path)
        write_user_target(self.output().path,
                          synapses,
                          name=self.mtype)

    def output(self):
        return LocalTarget('{}/user.target'.format(self.folder))


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
        write_sonata.write_nodes(load(self.input().path),
                                 self.output().path,
                                 self.node_population,
                                 self.mtype)

    def output(self):
        return LocalTarget(self._get_full_path_output(self.node_file_name))


class WriteSonataEdges(WriteSonata):
    """Write Sonata edges file to be parameterized with Spykfunc."""

    def requires(self):
        return self.clone(ReducePrune)

    def run(self):
        write_sonata.write_edges(load(self.input().path),
                                 self.output().path,
                                 self.edge_population)

    def output(self):
        return LocalTarget(self._get_full_path_output('nonparameterized_' + self.edge_file_name))


def _check_if_old_syntax(archive):
    """Check if old command format needs to be used with spykfunc and parquet-converters.

    New format is expected starting from archive/2021-07."""
    m = re.match(r'archive/(?P<year>\d+)-(?P<month>\d+)', archive)
    return m is not None and not (m.group('year') >= '2021' and m.group('month') >= '07')


class RunSpykfunc(WriteSonata):
    """Run spykfunc for the projections."""

    def requires(self):
        return self.clone(WriteSonataEdges), self.clone(WriteSonataNodes)

    def _parse_command(self):
        circuit_dir = os.path.dirname(self.circuit_config)
        spykfunc_dir = self.output().path
        edges = self.input()[0].path
        from_nodes = self.input()[1].path
        to_nodes = os.path.join(circuit_dir,
                                'sonata/networks/nodes/',
                                self.target_population,
                                'nodes.h5')
        cluster_dir = self._get_full_path_output('_sm_cluster')
        command = (f"module purge; module load {self.module_archive} spykfunc; "
                   f"unset SLURM_MEM_PER_NODE; unset SLURM_MEM_PER_GPU; unset SLURM_MEM_PER_CPU; "
                   f"sm_run -m 0 -w {cluster_dir} spykfunc --output-dir={spykfunc_dir} "
                   f"-p spark.master=spark://$(hostname):7077 "
                   f"--from {from_nodes} {self.node_population} "
                   f"--to {to_nodes} {self.target_population} "
                   f"--filters AddID,SynapseProperties ")

        if _check_if_old_syntax(self.module_archive):
            command += (f"--touches {edges} {self.edge_population} "
                        f"{self.recipe_path} "
                        f"{self.morphology_path} ")
        else:
            command += (f"'--recipe' {self.recipe_path} "
                        f"'--morphologies' {self.morphology_path} "
                        f"{edges} {self.edge_population} ")

        return command

    def run(self):
        try:
            subprocess.run(self._parse_command(), shell=True, check=True)
        except subprocess.CalledProcessError:
            if os.path.isdir(self.output().path):
                shutil.rmtree(self.output().path)
            raise

    def output(self):
        return LocalTarget(self._get_full_path_output('spykfunc'))


class RunParquetConverter(WriteSonata):
    """Run parquet converters for the spykfunc parquet files."""

    def requires(self):
        return self.clone(RunSpykfunc), self.clone(WriteSonataNodes)

    def _parse_command(self):
        circuit_dir = os.path.dirname(self.circuit_config)
        from_nodes = self.input()[1].path
        parquet_dir = os.path.join(self.input()[0].path, "circuit.parquet")
        parquet_glob = os.path.join(parquet_dir, "*.parquet")
        to_nodes = os.path.join(circuit_dir, 'sonata/networks/nodes/',
                                self.target_population, 'nodes.h5')
        edge_file_name = self.output().path
        command = (f"module purge; "
                   f"module load {self.module_archive} parquet-converters; "
                   f"parquet2hdf5 ")

        if _check_if_old_syntax(self.module_archive):
            command += (f"--format SONATA "
                        f"--from {from_nodes} {self.node_population} "
                        f"--to {to_nodes} {self.target_population} "
                        f"'-o' {edge_file_name} "
                        f"'-p' {self.edge_population} "
                        f"{parquet_glob} ")
        else:
            command += f"{parquet_dir} {edge_file_name} {self.edge_population}"

        return command

    def run(self):
        subprocess.run(self._parse_command(), shell=True, check=True)

    def output(self):
        return LocalTarget(self._get_full_path_output(self.edge_file_name))
