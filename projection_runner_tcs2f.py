# workaround for now forcing my old build, until this version mismatch issue with spatial indexer is sorted out
import sys
sys.path.insert(1,"/gpfs/bbp.cscs.ch/project/proj1/software/legacy-spatialindexer/lib")
import libFLATIndex
print "Indexer imported from: ", libFLATIndex


import projection_utility as proj
import os

#xml_file = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_input_generic_L4/thalamocorticalProjectionRecipe.xml"
#xml_file = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_input_generic_L4/thalamocorticalProjectionRecipe_1x7_0_TCs2f_7synsPerConn_os2p66.xml"

#xml_file = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_input_generic_L4/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os3p0.xml"

#xml_file = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os3p0.xml"
#xml_file = "/gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os3p0.xml"
xml_file = "/gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6.xml"
#xml_file = "/gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_POm/thalamocorticalProjectionRecipe_O1_TCs2f_os2p6.xml"


#xml_file = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_input_generic_L4/thalamocorticalProjectionRecipe_debug.xml"

#cfg_file = "/bgscratch/bbp/circuits/11.06.12/SomatosensoryCxS1-v4.lowerCellDensity.r135/1x7/merged_circuit/CircuitConfig"

#cfg_file = "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/1x7_0/merged_circuit/CircuitConfig"

#cfg_file = "/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/1x7_0/merged_circuit/CircuitConfig"

#cfg_file = "/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"
cfg_file = "/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/merged_circuit/CircuitConfig"


#cfg_file = "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"

composer = proj.ProjectionComposer(cfg_file, xml_file)
if not os.path.exists("out"):
    os.mkdir("out")
composer.write("out/", 8192)

# recover h5 write fail
"""
composer.proj_list[0].write_h5_file("out/", 'proj_nrn', 8192)


"""