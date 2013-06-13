import h5py
import numpy
import pylab
import scipy.stats
import scipy.interpolate
""""
A very simple set of functions implementing a kind of 'Thalamocortical S2F'
by only accepting connections with a certain min number of synapses.

"""


def thalamocortical_s2f(in_file, out_file, params):
    in_h5 = h5py.File(in_file,'r')
    out_h5 = h5py.File(out_file,'w')
    real_params = (parameter_estimation(params[0],params[1]), params[1])
    for k in in_h5.keys():
        data = numpy.array(in_h5[k])
        post_lids = numpy.unique(data[:,0],return_inverse=True)[1]
        unique_lids = numpy.unique(post_lids)
        lid_counts = numpy.histogram(post_lids, bins=numpy.hstack((unique_lids,unique_lids[-1]+1)))
        accepted = numpy.ones_like(post_lids)
        for t_id,count in zip(unique_lids, lid_counts):
            accepted[post_lids == t_id] = cutoff_func(count,params)
        
        if pylab.any(accepted):
            write_me = data[pylab.find(accepted),:]                
            out_h5.create_dataset(k, write_me.shape, ">f4", write_me)
    
def parameter_estimation(a, v):
    norm = scipy.stats.norm.cdf
    m = int(a)
    r = 1 - (a - m)
    tgt = m + 1
    x = pylab.linspace(tgt+2*v,m-2*v,100)
    f = norm(tgt,x,v)
    func = scipy.interpolate.interp1d(f,x)
    return func(r)    
    
def cutoff_func(x,params):
    norm = scipy.stats.norm.cdf
    return norm(x,params[0],params[1])>pylab.rand()
