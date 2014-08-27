import projection_utility as proj
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
efferent_h5_file = os.path.join("out", "proj_nrn_efferent.h5")
if not os.path.exists("out_s2f"):
    os.mkdir("out_s2f")
s2f_h5_file_efferent = os.path.join("out_s2f", "proj_nrn_efferent.h5")
s2f_h5_file_afferent = os.path.join("out_s2f", "proj_nrn.h5")

from tools.transpose_proj_nrn import transpose_projection
transpose_projection("out", "proj_nrn.h5.*", efferent_h5_file)
from tools.thalamocortical_s2f import thalamocortical_s2f
# Run s2f with either a target reduction factor or a target mean number of synapses per connection.
# Ideally, both yield statistically identical results
# thalamocortical_s2f('out/proj_nrn_efferent.h5','out/proj_nrn_efferent_s2f.h5',target_mean=tgt_mean)
thalamocortical_s2f(efferent_h5_file, s2f_h5_file_efferent, target_remove = 1 - 1/super_smpl_factor)

# Transpose efferent back to NRN
transpose_projection("out_s2f", "proj_nrn_efferent.h5", s2f_h5_file_afferent)

#Finally, split the result into many files:
from tools.split_synapse_files import split_nrn
split_nrn(s2f_h5_file_afferent, 8192)



