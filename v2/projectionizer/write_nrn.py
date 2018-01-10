'''tools for writing nrn_*.h5 files'''
import h5py
import numpy as np


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

    CONDUCTION_VELOCITY = 300.  # micron/ms, from original Projectionizer: InputMappers.py
    synapse_data[:, SynapseColumns.DELAY] = synapses['sgid_distance'].values / CONDUCTION_VELOCITY

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
            specification_conformity_check(synapse_data)

            h5.create_dataset('a%d' % gid, data=synapse_data)

            if efferent:
                N = len(synapses["syn_ids"])
                h5.create_dataset('a%d_afferentIndices' %
                                  gid, data=synapses["syn_ids"].values.reshape((N, 1)))


def specification_conformity_check(data):
    '''Check the nrn.h5 file specification'''
    assertions = (
        ('All Delays (column: {}) must be positive'.format(SynapseColumns.DELAY),
         data[:, SynapseColumns.DELAY] >= 0.),

        ('All offsets (column: {}) must be between 0 and 1'.format(SynapseColumns.OFFSET),
         (data[:, SynapseColumns.OFFSET] >= 0.) & (data[:, SynapseColumns.OFFSET] <= 1.)),

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
