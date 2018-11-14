'''tools for writing nrn_*.h5 files'''
import h5py
import numpy as np
import pandas as pd


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


def create_synapse_data(synapses, synapse_params, efferent, version=0):
    '''return numpy array for `synapses` with the correct parameters

    Args:
        synapses(pd.DataFrame): with the following columns:
            sgid, y, section_id, segment_id, offset

    Note: versions < 5 use ASE, and >=5 use NRRP
    '''
    synapse_count = len(synapses)
    synapse_data = np.zeros((synapse_count, 19), dtype=np.float)

    synapse_data[:, SynapseColumns.SGID] = synapses['tgid' if efferent else 'sgid'].values

    CONDUCTION_VELOCITY = 300.  # micron/ms, from original Projectionizer: InputMappers.py
    synapse_data[:, SynapseColumns.DELAY] = (synapses['sgid_path_distance'].values /
                                             CONDUCTION_VELOCITY)

    synapse_data[:, SynapseColumns.ISEC] = synapses['section_id'].values
    synapse_data[:, SynapseColumns.IPT] = synapses['segment_id'].values
    synapse_data[:, SynapseColumns.OFFSET] = synapses['synapse_offset'].values

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
    if version >= 5:
        synapse_data[:, SynapseColumns.ASE] = gamma('NRRP')
    else:
        synapse_data[:, SynapseColumns.ASE] = gamma('ASE')

    return synapse_data


def write_synapses(path, itr, synapse_params, efferent=False,
                   populate_synapse_data=create_synapse_data,
                   version=4):
    '''write synapses to nrn.h5 style file

    Args:
        path(str): path to file to output
        itr(tuple of (target GID, synapses), where synapses can be an iterable with
        DataFrame with colunms ['sgid', 'section_id', 'segment_id', 'synapse_offset'],
        If efferent==True, the column 'afferent_indices' must also be present.
    '''
    with h5py.File(path, 'w') as h5:
        info = h5.create_dataset('info', data=0)
        info.attrs['version'] = version
        info.attrs['numberOfFiles'] = 1
        for gid, synapses in itr:
            synapse_data = populate_synapse_data(synapses, synapse_params, efferent, version)
            specification_conformity_check(synapse_data)

            h5.create_dataset('a%d' % gid, data=synapse_data)

            if efferent:
                N = len(synapses['afferent_indices'])
                h5.create_dataset('a%d_afferentIndices' %
                                  gid, data=synapses['afferent_indices'].values.reshape((N, 1)))


def specification_conformity_check(data):
    '''Check the nrn.h5 file specification'''
    assertions = (
        ('All Delays (column: {}) must be positive'.format(SynapseColumns.DELAY),
         data[:, SynapseColumns.DELAY] >= 0.),

        ('All offsets (column: {}) must be positive'.format(SynapseColumns.OFFSET),
         (data[:, SynapseColumns.OFFSET] >= 0.)),

        ('GIDs (column: {}) must be positive'.format(SynapseColumns.SGID),
         data[:, SynapseColumns.SGID] >= 0),

        ('Synapse type (column: {}) must be positive'.format(SynapseColumns.SYNTYPE),
         data[:, SynapseColumns.SYNTYPE] >= 0),

        ('Depression SR recovery time constant (column: {}) must be positive'.format(
            SynapseColumns.D),
         data[:, SynapseColumns.D] >= 0.),

        ('Facilitation SR recovery time constant (column: {}) must be positive'.format(
            SynapseColumns.F),
         data[:, SynapseColumns.F] >= 0.),

        ('Decay time constant SR (column: {}) must be positive'.format(SynapseColumns.DTC),
         data[:, SynapseColumns.DTC] >= 0.),
    )

    for error_msg, assertion in assertions:
        assert np.all(assertion), error_msg


def write_synapses_summary(path, synapses):
    '''write synapses to nrn_summary.h5 style file

    Args:
        path(str): path to file to output
        per https://bbpteam.epfl.ch/project/spaces/display/BBPHPC/Synapses+-+nrn_summary.h5

    '''
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

    with h5py.File(path, 'w') as h5:
        for dataset, connections in summary.groupby("dataset"):
            data = np.zeros((len(connections["connecting"].values), 3), dtype=np.int)
            data[:, 0] = connections["connecting"].values
            data[:, 1] = connections["efferent"].values
            data[:, 2] = connections["afferent"].values
            h5.create_dataset('a%d' % dataset, data=data)


def write_user_target(output, synapses, name):
    '''write target file

    Args:
        output(path): path of file to create
        synapses(dataframe): synapses
        name(str): name of target
    '''
    with open(output, 'w') as fd:
        fd.write('Target Cell %s {\n' % name)
        for tgid in sorted(synapses.sgid.unique()):
            fd.write('    a{}\n'.format(tgid))
        fd.write('}\n')
