import h5py
import numpy
import pylab
import scipy.stats
from os import path
import glob
import progressbar as pb
from bluepy.targets.mvddb import Neuron, MType, EType, to_gid_list

""""
A very simple set of functions implementing a kind of 'Thalamocortical S2F'
by only accepting connections with a certain min number of synapses.

This is different to thalamocortical_s2f in that it is "ps" -> Pathway specific

"""


def thalamocortical_s2f(in_file, out_file, circuit_config, cutoff_var, target_mean = None, target_remove = None, filter_pre_keys=None):
    if not path.exists(in_file):
        in_files = glob.glob(in_file + '.')
        suffix = [x[len(in_file):] for x in in_files]
    else:
        in_files = [in_file]
        suffix = ['']

    circ = circuit_config
    # create class gid maps
    postsyn_class_gids = {}
    for mtype_id in circ.mvddb.mtype_id2name_map().keys():
        postsyn_class_gids[mtype_id] = set(circ.mvddb.select_gids(Neuron.mtype_id==mtype_id)) 

    if target_mean != None:
        cutoff_mean = parameter_from_mean(in_files, target_mean, postsyn_class_gids)
    elif target_remove != None:
        cutoff_mean = parameter_from_fraction_removed(in_files, target_remove, postsyn_class_gids, filter_pre_keys=filter_pre_keys)
    else:
        raise RuntimeError("Must provide one of two parameters: target_mean or target_remove")
    
    widgets = ['TC S2F: ', pb.Percentage(), ' ', pb.Bar(),
               ' ', pb.ETA()]
    
    for i, (f_in,suff) in enumerate(zip(in_files,suffix)):
        in_h5 = h5py.File(f_in,'r')
        out_h5 = h5py.File(out_file + suff)
        pbar = pb.ProgressBar(widgets=widgets, maxval=len(in_h5.keys())).start()
        for i,k in enumerate(in_h5.keys()):
            data = numpy.array(in_h5[k])
            gid_counts, post_gids, unique_gids = count_syns_connection(data, return_gids=True, return_uniques=True)        
            accepted = numpy.ones_like(post_gids)
            mtype_ids = [n.mtype_id for n in circ.mvddb.get_gids(post_gids)]
            for gid,mtype_id, count in zip(unique_gids, mtype_ids, gid_counts):
                accepted[post_gids == gid] = cutoff_func(count,(cutoff_mean[mtype_id], cutoff_var))
        
            if pylab.any(accepted):
                write_me = data[pylab.find(accepted),:]                
                out_h5.create_dataset(k, write_me.shape, ">f4", write_me)
            pbar.update(i+1)

        pbar.finish()
        in_h5.close()
        out_h5.close()
            
def parameter_from_fraction_removed(in_files, fraction, postsyn_class_gids, filter_pre_keys=None):    
    mtype_nsyn_samples = {}
    for name in in_files:
        in_h5 = h5py.File(name)
        for k in in_h5.keys():
            if filter_pre_keys and k not in filter_pre_keys:
                continue
            data = numpy.array(in_h5[k])
            syns_conn_samples, sample_gids = count_syns_connection(data,return_uniques=True)
            # loop over mtypes
            for class_name,class_gids in postsyn_class_gids.iteritems():                
                mtype_nsyn_samples.setdefault(class_name,[]).append(syns_conn_samples[numpy.array([x in class_gids for x in sample_gids])])
        in_h5.close()
    proposed_cutoff = {}
    for class_name,nsyn_samples in mtype_nsyn_samples.iteritems():
        # EM: are you sure its not 42?
        count_counts = numpy.histogram(numpy.hstack(nsyn_samples),range(1,52))[0]
        count_counts = [float((x+1)*y) for x,y in enumerate(count_counts)]
        cumulative_count = numpy.hstack((0,numpy.cumsum(count_counts)/sum(count_counts)))
        proposed_cutoff[class_name] = numpy.interp(fraction,cumulative_count,range(1,len(cumulative_count)+1))-1
    return proposed_cutoff

def parameter_from_mean(in_files,tgt_mn, postsyn_class_gids, filter_pre_keys=None):
    mtype_nsyn_samples = {}
    for name in in_files:
        in_h5 = h5py.File(name)
        for k in in_h5.keys():
            if filter_pre_keys and k not in filter_pre_keys:
                continue
            data = numpy.array(in_h5[k])
            syns_conn_samples, sample_gids = count_syns_connection(data,return_uniques=True)
            for class_name,class_gids in postsyn_class_gids.iteritems():                
                mtype_nsyn_samples.setdefault(class_name,[]).append(syns_conn_samples[numpy.array([x in class_gids for x in sample_gids])])
        in_h5.close()
    proposed_cutoff = {}
    for class_name,nsyn_samples in postsyn_class_gids.iteritems():
        proposed_cutoff[class_name] = tgt_mn - numpy.mean(nsyn_samples) + 0.5
    return proposed_cutoff
            
def count_syns_connection(data, return_gids=False, return_uniques=False):
    post_gids = data[:,0]    
    unique_gids = numpy.unique(post_gids)
    return_me = numpy.histogram(post_gids, bins=numpy.hstack((unique_gids,unique_gids[-1]+1)))[0]
    if return_gids:
        if return_uniques:
            return (return_me, post_gids, unique_gids)
        else:
            return (return_me, post_gids)
    else:
        if return_uniques:
            return (return_me, unique_gids)
        else:
            return return_me
    
def cutoff_func(x,params):
    norm = scipy.stats.norm.cdf
    return norm(x,params[0],params[1])>pylab.rand()
