from bbProjections import projection_utility as proj
import os

xml_file = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_input_generic_L4/thalamocorticalProjectionRecipe.xml"

#xml_file = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_input_generic_L4/thalamocorticalProjectionRecipe_debug.xml"

#cfg_file = "/bgscratch/bbp/circuits/11.06.12/SomatosensoryCxS1-v4.lowerCellDensity.r135/1x7/merged_circuit/CircuitConfig"

#cfg_file = "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/1x7_0/merged_circuit/CircuitConfig"

cfg_file = "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"

composer = proj.ProjectionComposer(cfg_file, xml_file)
if not os.path.exists("out"):
    os.mkdir("out")
composer.write("out/", 8192)

#For some projections we have to increase the mean number of synapses per connection. 
#We do that by creating a projection with higher density and removing connections with too few synapses.
#This is similar to the column s2f algorithm.
super_smpl_factor = 2.42
tgt_mean = 7.0

#CODE TO GENERATE EFFERENT NRN GOES HERE
# in dir where the proj_nrn.h5.* files reside:
python ~/src/bbp-user-ebmuller/experiments/thalamocortical_projection/plots/transpose_proj_nrn.py
#CODE TO MERGE NRN GOES HERE
python ~/src/bbp-user-ebmuller/SynMerge/TCSynMerge.py

from tools.thalamocortical_s2F import thalamocortical_s2f
#Run s2f with either a target reduction factor or a target mean number of synapses per connection.
#Ideally, both yield statistically identical results
thalamocortical_s2f('out/proj_nrn_efferent.h5','out/proj_nrn_efferent_s2f.h5',target_mean=tgt_mean)
thalamocortical_s2f('out/proj_nrn_efferent.h5','out/proj_nrn_efferent_s2f.h5',target_remove = 1 - 1/super_smpl_factor)

# Transpose efferent back to NRN
# Move to new dir, proj_nrn_efferent_s2f.h5 -> proj_nrn_efferent.h5
ln -s proj_nrn_efferent.h5 proj_nrn_efferent.h5.0
# Transpose efferent back to NRN
python ~/src/bbp-user-ebmuller/experiments/thalamocortical_projection/plots/transpose_proj_nrn_efferent.py

#Finally, split the result into many files:
from tools.split_synapse_files import split_nrn
split_nrn('out/proj_nrn_s2f.h5',8192)



