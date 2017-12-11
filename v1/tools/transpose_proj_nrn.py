import glob
import sys
import os

import progressbar as pb
import re

# override system h5py with h5py 2.0.1
# Taking this out for now since it refers to a home directory. I hope it is not needed...
# sys.path.insert(0,"/home/ebmuller/opt/lib/python2.6/site-packages")

import h5py
import numpy


def transpose_projection(root, pattern, out_file):
    files = glob.glob(os.path.join(root, pattern))
    # files = glob.glob("proj_nrn.h5.*")

    # sort in order of rank
    files.sort(key=lambda x: int(x.split('.h5.')[1]))
    widgets = ['Proj invert: ', pb.Percentage(), ' ', pb.Bar(),
               ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=len(files)).start()
    regex_gid = re.compile("a\d+\Z")
    d = {}

    for i, f in enumerate(files):
        h5 = h5py.File(f)
        for ds in h5:
            if not regex_gid.match(ds):
                continue
            syn_data = h5[ds].value
            gid = int(ds[1:])
            for syn in syn_data:
                # prepare to store in efferent syns dictionary
                eff_syns = d.setdefault(syn[0], [])
                # replace pre_gid with post_gid
                syn[0] = gid
                eff_syns.append(syn)

        h5.close()
        pbar.update(i+1)
    pbar.finish()

    widgets = ['Writing efferent file ...: ', pb.Percentage(), ' ', pb.Bar(),
               ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=len(d)).start()
    of = h5py.File(out_file, "w")
    for i, gid in enumerate(d):
        syns = numpy.array(d[gid])
        ds_name = "a%d" % gid
        ds = of.create_dataset(ds_name, data=syns, compression='gzip')
        pbar.update(i+1)
    of.close()
    pbar.finish()
