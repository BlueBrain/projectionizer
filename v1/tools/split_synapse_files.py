import h5py
import numpy

""" A very simple function to split any nrn.h5 file into several files.
"""

def split_nrn(in_file, number):
    in_h5 = h5py.File(in_file,'r')
    k = in_h5.keys()
    split_idx = numpy.linspace(0,len(k),number+1).astype(int)
    split_k = numpy.split(k, split_idx[1:-1])
    #import pdb; pdb.set_trace()
    for i,keys in enumerate(split_k):
        out_h5 = h5py.File(in_file + '.' + str(i),'w')
        for key in keys:
            write_me = in_h5[key]                
            out_h5.create_dataset(str(key), write_me.shape, ">f4", write_me)
            #out_h5.copy(in_h5[key], key)
        out_h5.close()
    in_h5.close()
