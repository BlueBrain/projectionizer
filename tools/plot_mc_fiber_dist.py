import bluepy
from bluepy.synapses import morph_stat_helper
import numpy
import pylab


"""Helper function that returns the locations of a random subsample of projection synapses.
Subset is sampled by randomly picking a number of targeted postsynaptic cells"""
def get_projection_synapse_locations(circ,proj,max_num_cells=1000,return_fac=False):
    m = morph_stat_helper.MorphStatHelper(circ)
    idx = numpy.nonzero([x.name == proj for x in circ.projections])[0]
    
    if len(idx) == 0:
        raise RuntimeError('Projection ' + proj + ' not found!')
    
    minicolumn_gids = numpy.array([x for x in circ.projections[idx].source])
    minicolumn_gids.sort()
    
    post_gids = m.synapse_property_from_gids(minicolumn_gids,[0],direction='EFFERENT', projection= proj)
    unique_post_gids = map(int,numpy.unique(numpy.vstack(post_gids)))
    numpy.random.shuffle(unique_post_gids)
    selected_post_gids = numpy.sort(unique_post_gids[:numpy.min((max_num_cells,len(unique_post_gids)))])
    
    syn_loc = m.synapse_position_from_gids(selected_post_gids,applyTranslation=True,applyRotation=True,projection=proj)
    pre_gids = m.synapse_property_from_gids(selected_post_gids,[0],projection=proj)
    if return_fac:
        return (syn_loc, float(len(unique_post_gids)) / float(len(selected_post_gids)))
    else:
        return (syn_loc, pre_gids)
    
"""Plots the distribution of the horizontal distances of synapses from the center of their minicolumns
"""
def plot_mc_fiber_dist(circ,proj,max_num_cells=1000):    
    idx = numpy.nonzero([x.name == proj for x in circ.projections])[0]
    
    if len(idx) == 0:
        raise RuntimeError('Projection ' + proj + ' not found!')
    
    minicolumn_gids = numpy.array([x for x in circ.projections[idx].source])
    minicolumn_gids.sort()
    minicolumn_coords = circ.mvd_minicolumns
    
    syn_loc, pre_gids = get_projection_synapse_locations(circ,proj, max_num_cells=max_num_cells)
    mc_ids = [numpy.searchsorted(minicolumn_gids, x) for x in pre_gids]
    rel_syn_loc = numpy.vstack([numpy.vstack(
                                             [l[[0,2]] - minicolumn_coords[i] for l,i in zip(loc,idx)]
                                             ) for loc,idx in zip(syn_loc,mc_ids)])
    rel_dist = numpy.sqrt(numpy.sum(numpy.power(rel_syn_loc,2),axis=1))
    
    H = numpy.histogram(rel_dist,bins=100)
    c_area = numpy.pi * numpy.power(H[1][1:],2) - numpy.pi * numpy.power(H[1][:-1],2)
    w = numpy.mean(numpy.diff(H[1]))
    normalize = lambda x: x.astype(float) / (numpy.sum(x) * w)
    
    pylab.bar(H[1][1:],normalize(H[0]/c_area))
    pylab.xlabel('Distance (um)')
    pylab.ylabel('Probability density')
    
"""Plots the y-profile of the absolute distance of projection synapses
"""
def plot_absolute_y_dist(circ,proj,max_num_cells=1000):
    syn_loc, fac = get_projection_synapse_locations(circ,proj, max_num_cells=max_num_cells,return_fac=True)
    syn_loc = numpy.vstack([numpy.vstack(x) for x in syn_loc])
    #syn_y = numpy.hstack([numpy.array([x[1] for x in xx]) for xx in syn_loc])
    
    H = numpy.histogramdd(syn_loc[:,[0,2,1]],bins=(20,20,40))    
    w = numpy.mean(numpy.diff(H[1][2]))
    v = numpy.prod([numpy.mean(numpy.diff(x)) for x in H[1]])
    data = fac * H[0].reshape(400,40).astype(float)/v
    data_mn = numpy.mean(data,axis=0)
    data_sd = numpy.std(data,axis=0)
    
    pylab.barh(H[1][2][:-1],data_mn,w)
    pylab.errorbar(data_mn,(H[1][2][:-1]+H[1][2][1:])/2,xerr=data_sd)
    pylab.ylabel('y-coordinate (um)')
    pylab.xlabel('Density (/um^3)')
    pylab.xlim((0,pylab.xlim()[1]))
    
    