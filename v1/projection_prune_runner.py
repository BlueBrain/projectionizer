import sys
import os
# add path to where this .py lives
sys.path.append(os.path.split(os.path.abspath(__file__))[0])

#super_smpl_factor = 2.42
tgt_mean = 7.0
cutoff_var = 1.0

#source_dir = "/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/1x7_0/merged_circuit/ncsThalamocortical_L4_oversample1p655/'out/proj_nrn_efferent.h5"

#python ~/src/bbp-user-ebmuller/SynMerge/TCSynMerge.py
#python ~/src/bbp-user-ebmuller/experiments/thalamocortical_projection/plots/transpose_proj_nrn.py

#python /bgscratch/bbp/users/ebmuller/Projectionizer-git/projection_prune_runner.py
#...

from tools.thalamocortical_s2f import thalamocortical_s2f
#Run s2f with either a target reduction factor or a target mean number of synapses per connection.
#Ideally, both yield statistically identical results

thalamocortical_s2f("proj_nrn_efferent.h5",'proj_nrn_efferent_s2f.h5', cutoff_var, target_mean=tgt_mean)

#thalamocortical_s2f('out/proj_nrn_efferent.h5','out/proj_nrn_efferent_s2f.h5',target_remove = 1 - 1/super_smpl_factor)

# Move to new dir, proj_nrn_efferent_s2f.h5 -> proj_nrn_efferent.h5
#ln -s proj_nrn_efferent.h5 proj_nrn_efferent.h5.0
# Transpose efferent back to NRN
#python ~/src/bbp-user-ebmuller/experiments/thalamocortical_projection/plots/transpose_proj_nrn_efferent.py

# spatial index:
#ln -s ../start.target .
#ln -s proj_nrn.h5 nrn.h5
#cp ../ncsThalamocortical_L4_tcS2F/CircuitConfig .
#edit nrnPath
#cp ../ncsThalamocortical_L4_tcS2F/build_synapse_index.py .
# edit build_synapse_index.py
# python build_synapse_index.py

#Finally, split the result into many files:
#from tools.split_synapse_files import split_nrn
#split_nrn('proj_nrn.h5',8192)

# python /bgscratch/bbp/users/ebmuller/Projectionizer-git/splitter.py
