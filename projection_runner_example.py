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
