'''Step 3: write nrn files
'''
import logging
import os
import traceback

import h5py
import numpy as np
import pandas as pd
from luigi import BoolParameter, FloatParameter, IntParameter
from luigi.local_target import LocalTarget

from projectionizer.step_1_assign import VirtualFibersNoOffset
from projectionizer.step_2_prune import ReducePrune
from projectionizer.utils import (CommonParams, FeatherTask, _write_feather,
                                  load)

L = logging.getLogger(__name__)


class SynapseColumns(object):
    '''columns index in nrn.h5 style file'''
    SGID = 0
    DELAY = 1

    ISEC = 2
    IPT = 3
    OFFSET = 4

    WEIGHT = 8  # G / CONDUCTANCE
    U = 9
    D = 10
    F = 11
    DTC = 12
    SYNTYPE = 13
    ASE = 17
    NEURITE_TYPE = 18

# from thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6_specific.xml
# synaptic parameters are mainly derived from Amitai, 1997;
# Castro-Alamancos & Connors 1997; Gil et al. 1997; Bannister et al. 2010 SR


def create_synapse_data(synapses, synapse_params, efferent):
    '''return numpy array for `synapses` with the correct parameters

    Args:
        synapses(pd.DataFrame): with the following columns:
            sgid, y, Section.Id, Segment.Id, location
    '''
    synapse_count = len(synapses)
    synapse_data = np.zeros((synapse_count, 19), dtype=np.float)

    synapse_data[:, SynapseColumns.SGID] = synapses['tgid' if efferent else 'sgid'].values

    # Note: this has to be > 0
    # (https://bbpteam.epfl.ch/project/issues/browse/NSETM-256?focusedCommentId=56509)
    # TODO: this needs to be a 'distance', for hex, 'y' makes sense, for 3D space, maybe less so
    CONDUCTION_VELOCITY = 300.  # micron/ms, from original Projectionizer: InputMappers.py
    synapse_data[:, SynapseColumns.DELAY] = synapses['y'].values / CONDUCTION_VELOCITY

    synapse_data[:, SynapseColumns.ISEC] = synapses['section_id'].values
    synapse_data[:, SynapseColumns.IPT] = synapses['segment_id'].values
    offset = synapses['location'].values * np.random.ranf(synapse_count)
    synapse_data[:, SynapseColumns.OFFSET] = offset

    def gamma(param):
        '''given `param`, look it up in SYNAPSE_PARAMS, return random pulls from gamma dist '''
        return np.random.gamma(shape=synapse_params[param][0],
                               scale=synapse_params[param][1],
                               size=synapse_count)

    synapse_data[:, SynapseColumns.WEIGHT] = gamma('gsyn')
    synapse_data[:, SynapseColumns.U] = gamma('Use')
    synapse_data[:, SynapseColumns.D] = gamma('D')
    synapse_data[:, SynapseColumns.F] = gamma('F')
    synapse_data[:, SynapseColumns.DTC] = gamma('DTC')
    synapse_data[:, SynapseColumns.SYNTYPE] = synapse_params['id']

    return synapse_data


def write_synapses(path, itr, synapse_params, efferent=False):
    '''write synapses to nrn.h5 style file

    Args:
        path(str): path to file to output
        itr(tuple of (target GID, synapses), where synapses can be an iterable with
        colums ['sgid', 'section_id', 'segment_id', 'location'] in order,
        or a pandas DataFrame with those columns
        If efferent==True, the column 'syn_ids' must also be present.
    '''
    with h5py.File(path, 'w') as h5:
        info = h5.create_dataset('info', data=0)
        info.attrs['version'] = 3
        info.attrs['numberOfFiles'] = 1
        for gid, synapses in itr:
            synapse_data = create_synapse_data(synapses, synapse_params, efferent)
            h5.create_dataset('a%d' % gid, data=synapse_data)

            if efferent:
                N = len(synapses["syn_ids"])
                h5.create_dataset('a%d_afferentIndices' %
                                  gid, data=synapses["syn_ids"].values.reshape((N, 1)))


def write_synapses_summary(path, itr):
    '''write synapses to nrn_summary.h5 style file

    Args:
        path(str): path to file to output
        itr(tuple of (target GID, synapses), where connections is a
        pandas DataFrame with columns ["connecting", "efferent", "afferent"]  as
        per https://bbpteam.epfl.ch/project/spaces/display/BBPHPC/Synapses+-+nrn_summary.h5

    '''
    with h5py.File(path, 'w') as h5:
        for dataset, connections in itr:
            data = np.zeros((len(connections["connecting"].values), 3), dtype=np.int)
            data[:, 0] = connections["connecting"].values
            data[:, 1] = connections["efferent"].values
            data[:, 2] = connections["afferent"].values
            h5.create_dataset('a%d' % dataset, data=data)


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
        afferent = synapses.groupby(["tgid", "sgid"]).count()["segment_id"].reset_index()
        afferent.columns = ["dataset", "connecting", "afferent"]
        summary = pd.merge(efferent, afferent, on=["dataset", "connecting"], how="outer")
        summary.fillna(0, inplace=True)
        summary["efferent"] = summary["efferent"].astype(np.int32)
        summary["afferent"] = summary["afferent"].astype(np.int32)
        try:
            write_synapses_summary(path=self.output().path,
                                   itr=summary.groupby("dataset"))
        except OSError as e:
            try:
                os.remove(self.output().path)
            except Exception:  # pylint: disable=broad-except
                pass
            traceback.print_exc()
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
            itr = load(self.input().path).groupby('sgid' if self.efferent else 'tgid')
            params = self.get_synapse_parameters()
            write_synapses(self.output().path, itr, params, efferent=self.efferent)
        except Exception as e:
            os.remove(self.output().path)
            traceback.print_exc()
            raise e

    def output(self):
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


class VirtualFibers(FeatherTask):
    '''Same as VirtualFibersNoOffset but with the sgid_offset'''

    def requires(self):  # pragma: no cover
        return self.clone(VirtualFibersNoOffset)

    def run(self):  # pragma: no cover
        fibers = load(self.input().path)
        fibers.index += self.sgid_offset  # pylint: disable=maybe-no-member
        _write_feather(self.output().path, fibers)
