'''Step 3: write nrn files
'''
import json
import logging
import os
import traceback

import numpy as np
import pandas as pd

from luigi import BoolParameter, FloatParameter, IntParameter
from luigi.contrib.simulate import RunAnywayTarget
from luigi.local_target import LocalTarget
from projectionizer.luigi_utils import CommonParams, CsvTask, JsonTask
from projectionizer.step_1_assign import VirtualFibersNoOffset
from projectionizer.step_2_prune import ChooseConnectionsToKeep, ReducePrune
from projectionizer.utils import load
from projectionizer.write_nrn import write_synapses, write_synapses_summary

L = logging.getLogger(__name__)


class WriteSummary(CommonParams):
    '''write proj_nrn_summary.h5'''

    def requires(self):  # pragma: no cover
        return self.clone(ReducePrune)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        synapses = load(self.input().path)
        efferent = synapses.groupby(["sgid", "tgid"]).count()["segment_id"].reset_index()
        efferent.columns = ["dataset", "connecting", "efferent"]
        afferent = synapses.groupby(["tgid", "sgid"]).count()[
            "segment_id"].reset_index()
        afferent.columns = ["dataset", "connecting", "afferent"]
        summary = pd.merge(efferent, afferent, on=["dataset", "connecting"],
                           how="outer")
        summary.fillna(0, inplace=True)
        summary["efferent"] = summary["efferent"].astype(np.int32)
        summary["afferent"] = summary["afferent"].astype(np.int32)

        try:
            write_synapses_summary(path=self.output().path,
                                   itr=summary.groupby("dataset"))
        except OSError as e:
            traceback.print_exc()
            try:
                os.remove(self.output().path)
            except OSError:
                pass
            raise e

    def output(self):
        return LocalTarget('{}/proj_nrn_summary.h5'.format(self.folder))


class WriteNrnH5(CommonParams):
    '''write proj_nrn.h5 or proj_nrn_efferent.h5'''
    efferent = BoolParameter()
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
            'Ase': get_gamma_parameters(self.ASE_mean, self.ASE_sigma),
        }

    def requires(self):  # pragma: no cover
        return self.clone(ReducePrune)

    def run(self):  # pragma: no cover
        try:
            # pylint thinks load() isn't returning a DataFrame
            # pylint: disable=maybe-no-member
            itr = load(self.input().path).groupby(
                'sgid' if self.efferent else 'tgid')
            params = self.get_synapse_parameters()
            write_synapses(self.output().path, itr,
                           params, efferent=self.efferent)
        except Exception as e:
            traceback.print_exc()
            try:
                os.remove(self.output().path)
            except OSError:
                pass
            raise e

    def output(self):  # pragma: no cover
        name = 'proj_nrn.h5' if not self.efferent else 'proj_nrn_efferent.h5'
        return LocalTarget('{}/{}'.format(self.folder, name))


class WriteUserTargetTxt(CommonParams):
    '''write user.target'''
    extension = 'target'

    def requires(self):  # pragma: no cover
        return self.clone(ReducePrune)

    def run(self):  # pragma: no cover
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        synapses = load(self.input().path)
        with self.output().open('w') as outfile:
            outfile.write('Target Cell proj_Thalamocortical_VPM_Source {\n')
            for tgid in sorted(synapses.sgid.unique()):
                outfile.write('    a{}\n'.format(tgid))
            outfile.write('}\n')

    def output(self):  # pragma: no cover
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


class SynapseCountPerConnectionL4PC(JsonTask):
    '''Compute the mean number of synapses per connection for L4 PC cells'''

    def requires(self):  # pragma: no cover
        return self.clone(ChooseConnectionsToKeep)

    def run(self):  # pragma: no cover
        connections = load(self.input().path)
        l4_pc_mtypes = ['L4_PC', 'L4_UPC', 'L4_TPC']

        # TODO: will be removed next commit
        # l4_pc_mtypes = [u'L4_BP', u'L4_BTC', u'L4_LBC', u'L4_MC', u'L4_SBC', u'L5_BP',
        #                 u'L5_BTC', u'L5_ChC', u'L5_DBC', u'L5_LBC', u'L5_MC', u'L5_NBC',
        #                 u'L5_SBC', u'L5_STPC', u'L5_TTPC1', u'L5_TTPC2', u'L5_UTPC',
        #                 u'L6_BP', u'L6_BPC', u'L6_BTC', u'L6_ChC', u'L6_DBC', u'L6_IPC',
        #                 u'L6_LBC', u'L6_MC', u'L6_NBC', u'L6_SBC', u'L6_TPC_L1',
        #                 u'L6_TPC_L4', u'L6_UTPC']

        # pylint: disable=maybe-no-member
        l4_pc_cells = connections[(connections.mtype.isin(l4_pc_mtypes)) &
                                  (connections.kept)]
        mean = l4_pc_cells.loc[:, '0'].mean()
        if np.isnan(mean):
            raise Exception('SynapseCountPerConnectionL4PC returned NaN')
        with self.output().open('w') as outputf:
            json.dump({'result': mean}, outputf)


class WriteAll(CommonParams):
    """Run all write tasks"""

    def requires(self):  # pragma: no cover
        return [self.clone(WriteNrnH5, efferent=True),
                self.clone(WriteNrnH5, efferent=False),
                self.clone(WriteSummary),
                self.clone(WriteUserTargetTxt),
                self.clone(SynapseCountPerConnectionL4PC)]

    def run(self):  # pragma: no cover
        self.output().done()

    def output(self):  # pragma: no cover
        return RunAnywayTarget(self)
