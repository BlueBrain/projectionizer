import h5py
import numpy
import pylab
""""
A very simple set of functions implementing a kind of 'Thalamocortical S2F'
by only accepting connections with a certain min number of synapses.

"""


def thalamocortical_s2f(in_file, out_file, params):
    in_h5 = h5py.File(in_file,'r')
    out_h5 = h5py.File(out_file,'w')
    for k in in_h5.keys():
        data = numpy.array(in_h5[k])
        post_lids = numpy.unique(data[:,0],return_inverse=True)[1]
        unique_lids = numpy.unique(post_lids)
        lid_counts = numpy.histogram(post_lids, bins=numpy.hstack((unique_lids,unique_lids[-1]+1)))
        accepted = numpy.ones_like(post_lids)
        for t_id,count in zip(unique_lids, lid_counts):
            accepted[post_lids == t_id] = cutoff_func(count,params)
        
        write_me = data[pylab.find(accepted),:]        
        out_h5.create_dataset(k, write_me.shape, ">f4", write_me)
    
    
def cutoff_func(x,params):
    raise NotImplementedError