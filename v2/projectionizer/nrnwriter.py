'''helper to write nrn.h5 files'''
import pandas as pd
import numpy as np
import h5py


def write_synapses(path, itr, create_synapse_data):
    '''write synapses to nrn.h5 style file

    Args:
        path(str): path to file to output
        itr(tuple of (target GID, df synapses)): usually a results of a synapses.groupby('tgid')
        create_synapse_data(function(above grouped DataFrames) -> Nx19 numpy array of synapses
        parameters): populate nrn file with parameters
    '''
    with h5py.File(path, 'w') as h5:
        for tgid, synapses in itr:
            synapse_data = create_synapse_data(synapses)
            h5.create_dataset('a%d' % tgid, data=synapse_data)
