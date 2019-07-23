'''Step 3: write nrn files
'''
import json
import logging
import os
import traceback
from functools import partial

import numpy as np

from luigi import FloatParameter, IntParameter, Parameter, DictParameter
from luigi.local_target import LocalTarget
from projectionizer.luigi_utils import CommonParams, CsvTask, JsonTask, RunAnywayTargetTempDir
from projectionizer.step_1_assign import VirtualFibersNoOffset
from projectionizer.step_2_prune import ChooseConnectionsToKeep, ReducePrune
from projectionizer.utils import load, ignore_exception
from projectionizer import write_nrn, write_syn2

L = logging.getLogger(__name__)


class WriteSummary(CommonParams):
    '''write proj_nrn_summary.h5'''

    def requires(self):
        return self.clone(ReducePrune)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        synapses = load(self.input().path)
        try:
            write_nrn.write_synapses_summary(path=self.output().path, synapses=synapses)
        except OSError as e:
            traceback.print_exc()
            with ignore_exception(OSError):
                os.remove(self.output().path)
            raise e

    def output(self):
        return LocalTarget('{}/proj_nrn_summary.h5'.format(self.folder))


class WriteNrnH5(CommonParams):
    '''write proj_nrn.h5'''
    synapse_type = IntParameter()
    gsyn_mean = FloatParameter()
    gsyn_sigma = FloatParameter()

    use_mean = FloatParameter()
    use_sigma = FloatParameter()

    D_mean = FloatParameter()
    D_sigma = FloatParameter()

    F_mean = FloatParameter()
    F_sigma = FloatParameter()

    DTC_mean = FloatParameter()
    DTC_sigma = FloatParameter()

    ASE_mean = FloatParameter()
    ASE_sigma = FloatParameter()

    def get_synapse_parameters(self):
        '''get the synapses paramaters based on config'''
        def get_gamma_parameters(mn, sd):
            '''transform mean/sigma parameters as per original projectionizer code'''
            return ((mn / sd) ** 2, (sd ** 2) / mn)  # k, theta or shape, scale

        return {
            'id': self.synapse_type,
            'gsyn': get_gamma_parameters(self.gsyn_mean, self.gsyn_sigma),
            'Use': get_gamma_parameters(self.use_mean, self.use_sigma),
            'D': get_gamma_parameters(self.D_mean, self.D_sigma),
            'F': get_gamma_parameters(self.F_mean, self.F_sigma),
            'DTC': get_gamma_parameters(self.DTC_mean, self.DTC_sigma),
            'ASE': get_gamma_parameters(self.ASE_mean, self.ASE_sigma),
        }

    def requires(self):
        return self.clone(ReducePrune)

    def run(self):
        try:
            # pylint thinks load() isn't returning a DataFrame
            # pylint: disable=maybe-no-member
            itr = load(self.input().path).groupby('tgid')
            params = self.get_synapse_parameters()
            write_nrn.write_synapses(self.output().path, itr, params)
        except Exception as e:
            traceback.print_exc()
            with ignore_exception(OSError):
                os.remove(self.output().path)
            raise e

    def output(self):
        name = 'proj_nrn.h5'
        return LocalTarget('{}/{}'.format(self.folder, name))


class WriteUserTargetTxt(CommonParams):
    '''write user.target'''
    target_name = Parameter('proj_Thalamocortical_VPM_Source')

    def requires(self):
        return self.clone(ReducePrune)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        synapses = load(self.input().path)
        write_nrn.write_user_target(self.output().path,
                                    synapses,
                                    name=self.target_name)

    def output(self):
        return LocalTarget('{}/user.target'.format(self.folder))


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


class WriteNrnH5Efferent(CommonParams):  # pragma: no cover
    '''write proj_nrn_efferent.h5'''
    def requires(self):
        return self.clone(WriteNrnH5)

    def run(self):
        write_nrn.rewrite_synapses_efferent(self.input().path,
                                            self.output().path)

    def output(self):
        name = 'proj_nrn_efferent.h5'
        return LocalTarget('{}/{}'.format(self.folder, name))


class WriteSyn2(CommonParams):
    '''write proj_nrn.syn2'''
    synapse_parameters = DictParameter()
    # micron/ms, from original Projectionizer: InputMappers.py
    conduction_velocity = FloatParameter(default=300.)

    def requires(self):
        return self.clone(ReducePrune)

    def run(self):
        try:
            # pylint thinks load() isn't returning a DataFrame
            # pylint: disable=maybe-no-member
            syns = load(self.input().path)
            syns['offset'] = syns['synapse_offset']

            # currently only support a single synapse type
            syns['synapse_type_name'] = 0

            syns['delay'] = (syns['sgid_path_distance'].values / self.conduction_velocity)

            synapse_data_creator = partial(write_syn2.create_synapse_data,
                                           synapse_data=self.synapse_parameters)
            write_syn2.write_synapses(syns,
                                      self.output().path,
                                      synapse_data_creator)
        except Exception as e:
            traceback.print_exc()
            with ignore_exception(OSError):
                os.remove(self.output().path)
            raise e

    def output(self):
        name = 'proj_nrn.syn2'
        return LocalTarget('{}/{}'.format(self.folder, name))


class WriteAll(CommonParams):  # pragma: no cover
    """Run all write tasks"""
    output_type = Parameter(default='nrn')

    def requires(self):
        output_type = str(self.output_type).lower()
        if output_type == 'nrn':
            return [self.clone(WriteNrnH5Efferent),
                    self.clone(WriteSummary),
                    self.clone(WriteUserTargetTxt),
                    self.clone(SynapseCountPerConnectionTarget),
                    ]
        elif output_type == 'syn2':
            return [self.clone(WriteSyn2),
                    ]

        raise Exception('unknown synapse output type: %s' % self.output_type)

    def run(self):
        self.output().done()

    def output(self):
        return RunAnywayTargetTempDir(self, base_dir=self.folder)
