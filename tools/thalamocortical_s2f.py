import h5py
import numpy
import pylab
import scipy.stats

""""
A very simple set of functions implementing a kind of 'Thalamocortical S2F'
by only accepting connections with a certain min number of synapses.

"""


def thalamocortical_s2f(in_file, out_file, cutoff_var, target_mean = None, target_remove = None):
    in_h5 = h5py.File(in_file,'r')
    out_h5 = h5py.File(out_file,'w')    
    if target_mean != None:
        cutoff_mean = parameter_from_mean(in_h5, target_mean)
    elif target_remove != None:
        cutoff_mean = parameter_from_fraction_removed(in_h5, target_remove)
    else:
        raise RuntimeError("Must provide one of two parameters: target_mean or target_remove")
    
    for k in in_h5.keys():
        data = numpy.array(in_h5[k])
        lid_counts, post_lids, unique_lids = count_syns_connection(data, return_lids=True, return_uniques=True)        
        accepted = numpy.ones_like(post_lids)
        for t_id,count in zip(unique_lids, lid_counts):
            accepted[post_lids == t_id] = cutoff_func(count,(cutoff_mean, cutoff_var))
        
        if pylab.any(accepted):
            write_me = data[pylab.find(accepted),:]                
            out_h5.create_dataset(k, write_me.shape, ">f4", write_me)
            
def parameter_from_fraction_removed(in_h5, fraction):
    lid_counts = []
    for k in in_h5.keys():
        data = numpy.array(in_h5[k])
        lid_counts.extend(count_syns_connection(data))
    count_counts = numpy.histogram(lid_counts,range(1,52))[0]
    count_counts = [float((x+1)*y) for x,y in enumerate(count_counts)]
    cumulative_count = numpy.hstack((0,numpy.cumsum(count_counts)/sum(count_counts)))
    return numpy.interp(fraction,cumulative_count,range(1,len(cumulative_count)+1))-1

def parameter_from_mean(in_h5,tgt_mn):
    lid_counts = []
    for k in in_h5.keys():
        data = numpy.array(in_h5[k])
        lid_counts.extend(count_syns_connection(data))
    return tgt_mn - numpy.mean(lid_counts) + 0.5
            
def count_syns_connection(data, return_lids=False, return_uniques=False):
    post_lids = numpy.unique(data[:,0],return_inverse=True)[1]
    unique_lids = numpy.unique(post_lids)
    return_me = numpy.histogram(post_lids, bins=numpy.hstack((unique_lids,unique_lids[-1]+1)))[0]
    if return_lids:
        if return_uniques:
            return (return_me, post_lids, unique_lids)
        else:
            return (return_me, post_lids)
    else:
        if return_uniques:
            return (return_me, unique_lids)
        else:
            return return_me
    
def cutoff_func(x,params):
    norm = scipy.stats.norm.cdf
    return norm(x,params[0],params[1])>pylab.rand()
