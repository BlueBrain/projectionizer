import sys
from lxml import etree
import numpy
import bluepy
from matplotlib import pyplot as plt

#proj_recipe_path = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os3p0.xml"

proj_recipe_path = "/gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe.xml"
#proj_recipe_path = "/gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_POm/thalamocorticalProjectionRecipe_O1_TCs2f_os2p6.xml"
#proj_recipe_path = "/gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6_specific.xml"
#proj_recipe_path = "/gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_POm/thalamocorticalProjectionRecipe.xml"


#proj_recipe_path = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Corticocortical_SIItoSI/corticocorticalProjectionRecipe.xml"
#proj_recipe_path = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_input_generic_L4/thalamocorticalProjectionRecipe.xml"

#proj_recipe_path = "/home/ebmuller/src/bbp_svn_recipe/Projection_Recipes/Thalamocortical_input_generic_L1/thalamocorticalProjectionRecipe.xml"

#circuit_path = "/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"
circuit_path = "/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/merged_circuit/CircuitConfig"



def plot_volume(v, c):
    bin_heights  = v.xpath("DensityProfile/bin/@height")
    bin_heights = numpy.array([float(x) for x in bin_heights])

    density_bins = numpy.zeros((len(bin_heights),2))
    bin_heights = numpy.hstack((0,(bin_heights[0:-1] + bin_heights[1:])/2,1))

    density_bins[:,0] = bin_heights[0:-1]
    density_bins[:,1] = bin_heights[1:]

    density_values  = v.xpath("DensityProfile/bin/@density")
    density_values = numpy.array([float(x) for x in density_values])

    # params to calibrate relative to absolute positions
    ymin_layer = int(v.xpath("Boundaries/Boundary[@which='y_min']/@id")[0])

    ymax_layer = int(v.xpath("Boundaries/Boundary[@which='y_max']/@id")[0])

    ymin_frac = float(v.xpath("Boundaries/Boundary[@which='y_min']/@rel")[0])

    ymax_frac = float(v.xpath("Boundaries/Boundary[@which='y_max']/@rel")[0])

    def compute_height(layer, frac):
        tops, bottoms, mids, ids = c.get_layers()
        thicks = tops-bottoms
        return bottoms[layer-1]+frac*thicks[layer-1]

    ymin = compute_height(ymin_layer, ymin_frac)
    ymax = compute_height(ymax_layer, ymax_frac)




    #p_bins = numpy.concatenate((density_bins[:,0], density_bins[-1,1]))
    db = density_bins[0,1]-density_bins[0,0]
    p_bins = numpy.hstack((density_bins[0,0]-db, density_bins[:,0], density_bins[-1,1]))
    p_bins_abs = p_bins*(ymax-ymin) + ymin
    #plot(p_bins, numpy.hstack((0, density_values, 0)), 'b-', ls='steps-post')
    plt.plot(numpy.hstack((0, density_values, 0))*1000, p_bins_abs, 'b-', ls='steps')


def plot_yprofile(circuit_path, proj_recipe_path):
    c = bluepy.Circuit(circuit_path)

    proj_recipe = etree.parse(proj_recipe_path)

    c.plot_layers()

    for v in proj_recipe.xpath("//Projections/Projection/Volume"):
        plot_volume(v, c)

    plt.axis([0, 100, -50, 2050])
    plt.xticks(numpy.arange(0,100,10), [str(x) for x in numpy.arange(0,100,10)])
    plt.xlabel("synapse density [1e-3 / um^3]")


if __name__=="__main__":

    if len(sys.argv)==3:
        circuit_path = sys.argv[1]
        proj_recipe_path = sys.argv[2]
    else:
        print "Usage: recipe_yprofile_plotter.py circuit_path proj_recipe_path"
        sys.exit()


    plot_yprofile(circuit_path, proj_recipe_path)
