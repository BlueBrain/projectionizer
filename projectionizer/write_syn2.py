'''tools for writing syn2 files'''
import logging
import numpy as np
import h5py


L = logging.getLogger(__name__)

DEFAULT_GROUP = '/synapses/default/properties'


# from https://github.com/adevress/syn2_spec/blob/master/spec_synapse_v2.md
DATASETS_TYPES = {'connected_neurons_pre': 'i8',
                  'connected_neurons_post': 'i8',

                  'afferent_center_x': 'f',
                  'afferent_center_y': 'f',
                  'afferent_center_z': 'f',

                  'delay': 'f',
                  'conductance': 'f',
                  'u_syn': 'f',
                  'depression_time': 'f',
                  'facilitation_time': 'f',
                  'decay_time': 'f',

                  # guessing on this, not in syn2 spec
                  'n_rrp_vesicles': 'i8',

                  'syn_type_id': 'i8',

                  'morpho_section_id_post': 'i8',
                  # guessing on this, not in syn2 spec
                  'morpho_segment_id_post': 'i8',
                  'morpho_offset_segment_post': 'f',

                  'conductance_scale_factor': 'f',
                  'u_hill_coefficient': 'f',
                  }

# physiology properties, dispached to _distribute_param
DATASET_PHYSIOLOGY_MAP = {
    'conductance': 'gsyn',
    'u_syn': 'Use',
    'depression_time': 'D',
    'facilitation_time': 'F',
    'decay_time': 'dtc',
    'n_rrp_vesicles': 'nrrp',
    'conductance_scale_factor': 'conductance_scale_factor',
    'u_hill_coefficient': 'u_hill_coefficient',
}

# properties that can be directly copied from DataFrame to output
DATASETS_DIRECT_MAP = {
    'afferent_center_x': 'x',
    'afferent_center_y': 'y',
    'afferent_center_z': 'z',
    'morpho_section_id_post': 'section_id',
    'morpho_segment_id_post': 'segment_id',
    'morpho_offset_segment_post': 'offset',
    'delay': 'delay',
}

# The gaussian truncation was revised to only be one standard deviation:
#  https://bbpteam.epfl.ch/project/issues/browse/NCX-246?focusedCommentId=8468
#  &page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel
#  #comment-84681
TRUNCATED_MAX_STDDEV = 1.

OPTIONAL_PARAMETERS = [
    'conductance_scale_factor',
    'u_hill_coefficient',
]


class OptionalParameterException(Exception):
    '''Raised if optional parameter not given.'''


def _truncated_gaussian(mean, std, count, truncated_max_stddev=TRUNCATED_MAX_STDDEV):
    '''create a truncated gaussian: sampled values are between 0 and truncated_max_stddev * stddev

    Note: definition of truncated gaussian
     https://bbpteam.epfl.ch/project/issues/browse/NCX-169?focusedCommentId=8208
     &page=com.atlassian.jira.plugin.system.issuetabpanels%3Acomment-tabpanel
     #comment-82081
    '''
    data = np.random.normal(mean, std, size=count)
    rejected = np.nonzero((np.abs(data - mean) > truncated_max_stddev * std) |
                          (data <= 0.))
    while len(rejected[0]):
        data[rejected] = np.random.normal(mean, std, size=len(rejected[0]))
        rejected = np.nonzero((np.abs(data - mean) > truncated_max_stddev * std) |
                              (data <= 0.))
    return data


def _distribute_param(df, synapse_data, prop, dtype):
    '''paramters with distributions: create random numbers'''
    ret = np.empty(len(df), dtype=dtype)
    for synapse_type_name, frame in df.reset_index().groupby('synapse_type_name'):
        dist = synapse_data['type_' + str(int(synapse_type_name))]

        if prop in OPTIONAL_PARAMETERS and DATASET_PHYSIOLOGY_MAP[prop] not in dist['physiology']:
            raise OptionalParameterException('Optional parameter not given.')

        dist = dist['physiology'][DATASET_PHYSIOLOGY_MAP[prop]]['distribution']
        assert dist['name']\
            in ('uniform_int', 'truncated_gaussian', 'fixed_value'),\
            'unknown distribution'

        if dist['name'] == 'uniform_int':
            low, high = dist['params']['min'], dist['params']['max']
            ret[frame.index] = np.random.random_integers(low, high, size=len(frame))
        elif dist['name'] == 'truncated_gaussian':
            mean, std = dist['params']['mean'], dist['params']['std']
            ret[frame.index] = _truncated_gaussian(mean, std, len(frame))
        elif dist['name'] == 'fixed_value':
            ret[frame.index] = dist['params']['value']
    return ret


def create_synapse_data(syn2_property_name, dtype, df, synapse_data):
    '''create concrete values for syn2_property_name for all proto-synapses in `df`

    Args:
        syn2_property_name(str): syn2 property name to be populated
        dtype(np.dtype): dtype to be returned
        df(pd.DataFrame): with columns sgid, tgid, x, y, z, section_id,
        segment_id, segment_offset
        synapse_data(dict): parameters used to populate synapses

    Returns:
        np.array with len(df)
    '''
    if syn2_property_name == 'connected_neurons_pre':
        ret = df.sgid.values.astype(dtype) - 1  # Note: syn2 is 0 indexed
    elif syn2_property_name == 'connected_neurons_post':
        ret = df.tgid.values.astype(dtype) - 1  # Note: syn2 is 0 indexed
    elif syn2_property_name == 'syn_type_id':
        ret = 120 * np.ones(len(df), dtype=dtype)  # TODO: get real syn type?
    elif syn2_property_name in DATASETS_DIRECT_MAP:
        ret = df[DATASETS_DIRECT_MAP[syn2_property_name]].values.astype(dtype)
    elif syn2_property_name in DATASET_PHYSIOLOGY_MAP:
        ret = _distribute_param(df, synapse_data, syn2_property_name, dtype)

    return ret


def write_synapses(syns, output, synapse_data_creator, datasets_types=None):
    '''Write the syn2 file for `syns`

    Args:
        syns(DataFrame): Synapses, must have ['tgid', 'sgid'], as well as
        all the values needed by synapse_data_creator to create the parameters
        for the keys in datasets_types
        output(str): path to write hdf5 file
        synapse_data_creator(callable):
        datasets_types(dict): dataset names -> np.dtype; the names with which
        synapse_data_creator is called are the keys of this dictionary

    Note: The technique used in write_nrn:write_synapses can't be used:
    the syn2 format has a monolithic dataset instead of a sub-dataset per
    gid.  This dataset needs to be sorted by (tgid, sgid) for the lookup
    to work in neurodamus
    '''
    if datasets_types is None:
        datasets_types = DATASETS_TYPES

    syns.sort_values(['tgid', 'sgid'], inplace=True)

    with h5py.File(output, 'w') as h5:
        properties = h5.create_group(DEFAULT_GROUP)
        for name, dtype in datasets_types.items():
            L.debug('Writing %s[%s]', output, name)
            try:
                properties.create_dataset(name,
                                          data=synapse_data_creator(name, dtype, syns))
            except OptionalParameterException:
                L.info('Optional parameter %s not given', name)
