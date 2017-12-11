import sys
import numpy
import matplotlib.pyplot as plt
import bluepy
from bluepy.targets.mvddb import Neuron, MType, EType, to_gid_list



#c = bluepy.Circuit("/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/1x7_0/merged_circuit/CircuitConfig")
#center_target = "mc3_Column"
#nsyns_per_fiber_gids = gid_offset+arange(930, 1240)

#c = bluepy.Circuit("/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig")

circuit_path = "/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/merged_circuit/CircuitConfig"

center_target = "mc2_Column"
#nsyns_per_fiber_gids = gid_offset+arange(620, 930)
#nsyns_per_fiber_gids = gid_offset+arange(620, 720)
#nsyns_per_fiber_gids = gid_offset+arange(620, 630)

def count_unique(keys):
    # using this recipe: 
    # http://stackoverflow.com/questions/10741346/numpy-frequency-counts-for-unique-values-in-an-array
    if len(keys)==0:
        return [],[]
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

def count_unique2(keys):
    d = {}
    if len(keys)==0:
        return {}
    for key in keys:
        d[key] = d.get(key,0)+1
    return d

def count_unique3(keys):
    d = count_unique2(keys)
    a = numpy.array(list(d.iteritems()))
    return a[:,0], a[:,1]


def plot_mean_spc(circuit_path):
    print "Circuit: %s" % circuit_path
    c = bluepy.Circuit(circuit_path)

    msh = c.morph_stat_helper

    #p = c.projections.Thalamocortical_input_VPM_tcS2F_3p0_ps
    #p = c.projections.Thalamocortical_input_VPM_tcS2F_2p6_ps
    p = c.projections.Thalamocortical_input_VPM

    #p = c.projections.Thalamocortical_input_generic_L4_tcS2F_3p0\
    #p = c.projections.Thalamocortical_input_generic_VPM_oversample3p0
    #p = c.projections.Thalamocortical_input_generic_L4_oversample3p0

    fiber_gids = sorted(p.source)[0]+numpy.arange(620, 630)

    post_gids = msh.synapse_property_from_gids(fiber_gids, [0], direction='EFFERENT', projection=p.name)

    num_syns = [len(x) for x in post_gids]
    # num syns/conn

    target_gid_groups = {'All':c.get_target(center_target),
                         'L4_EXC':c.mvddb.select_gids(Neuron.hyperColumn==2, MType.synapse_class=="EXC", Neuron.layer==4),
                         'L5_EXC':c.mvddb.select_gids(Neuron.hyperColumn==2, MType.synapse_class=="EXC", Neuron.layer==5),
                         'L4_INH':c.mvddb.select_gids(Neuron.hyperColumn==2, MType.synapse_class=="INH", Neuron.layer==4)}

    groups_nsyns_conn = {}
    for target_group,column_gids in target_gid_groups.iteritems(): 
        print "Group: %s" % target_group
        sys.stdout.flush()
        num_syns_conn =[]
        for gids in post_gids:
            #ugids, counts = count_unique(gids.flatten())
            #num_syns_conn+=list(counts)
            #gids_flat = gids.flatten()
            #gids_in_column = gids_flat.intersect(column_gids)

            gids = [int(gid) for gid in gids if int(gid) in column_gids]
            counts = count_unique2(gids)
            num_syns_conn+=list(counts.values())
        groups_nsyns_conn[target_group] = num_syns_conn

    plt.figure(facecolor='white')
    for target_group,num_syns_conn in groups_nsyns_conn.iteritems():
        h,bins = numpy.histogram(num_syns_conn, normed=True, bins=numpy.arange(0,30))
        plt.plot(bins,numpy.hstack((h,[0])),ls='steps-post', lw=2, label="%s, %f syns/conn" % (target_group, numpy.mean(num_syns_conn)))
        print "Group: %s, nsyns_per_conn=%f" % (target_group, numpy.mean(num_syns_conn))
    plt.xlabel("# syns/conn")
    plt.ylabel('a.u.')
    plt.legend(prop={'size':8})

    print "mean syns/conn = %f" % (numpy.mean(num_syns_conn),)


    ds1 = numpy.array(groups_nsyns_conn["L4_EXC"])
    ds2 = numpy.array(groups_nsyns_conn["L5_EXC"])

    n = min(len(ds1), len(ds2))

    ds = ds1[:n]
    bins = numpy.arange(0,30,2)
    h,bins = numpy.histogram(ds, bins=bins)
    db = bins[1]-bins[0]
        #plot(hstack(([bins[0]-db],bins)),hstack(([0],h,[0])),ls='steps-post', lw=2, label="%s\n, %f$\pm$%f, md=%f" %(f[:-3], mean(psps), std(psps),median(psps) ))
    plt.bar(bins+db*0.15, numpy.hstack((h,[0])), color='b', width=db*0.3,label="L4_EXC %f$\pm$%f, md=%f" %(numpy.mean(ds), numpy.std(ds), numpy.median(ds)))
    plt.errorbar(numpy.mean(ds), numpy.max(h*0.9), xerr=numpy.std(ds), fmt='bo', ms=8)

    ds = ds2[:n]
    h2,bins = numpy.histogram(ds, bins=bins)
    plt.bar(bins+db*0.55, numpy.hstack((h2,[0])), color='r', width=db*0.3, label="L5_EXC %f$\pm$%f, md=%f" %(numpy.mean(ds), numpy.std(ds), numpy.median(ds)))
    plt.errorbar(numpy.mean(ds), numpy.max(h2*0.9), xerr=numpy.std(ds), fmt='ro', ms=8)
    plt.legend(prop={'size':8})
    plt.show()



if __name__=="__main__":

    if len(sys.argv)==2:
        circuit_path = sys.argv[1]
    else:
        print "Usage: syn_mean_spc.py circuit_path"
        sys.exit()

    plot_mean_spc(circuit_path)


