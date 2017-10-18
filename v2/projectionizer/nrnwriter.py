import pandas as pd
import numpy as np
import h5py


def write_synapses(path, itr, create_synapse_data):
    '''write synapses to nrn.h5 style file

    Args:
        path(str): path to file to output
        itr(tuple of (target GID, synapses)), where synapses can be an iterable with
        columns ['sgid', 'section_id', 'segment_id', 'location'] in order,
        or a pandas DataFram with those columns
        create_synapse_data(function(Nx4 numpy array of [sgid, sec_id, seg_id, loc]) ->
        Nx19 numpy array of synapses parameters)
    '''
    with h5py.File(path, 'w') as h5:
        for tgid, synapse_iter in itr:
            if isinstance(synapse_iter, pd.DataFrame):
                columns = ['tgid', 'sgid', 'Section.ID', 'Segment.ID', 'location']
                synapses = synapse_iter[columns].values
            else:
                synapses = np.array([syn for syn in synapse_iter])
                if len(synapses.shape) != 2:
                    print 'bad tgid: %d' % tgid
                    continue
            synapse_data = create_synapse_data(synapses)
            h5.create_dataset('a%d' % tgid, data=synapse_data)
