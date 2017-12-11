import sys
import numpy
from matplotlib import pyplot as plt

import bluepy
from tools import recipe_yprofile_plotter



#circuit_path = "/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/merged_circuit/ncsThalamocortical_VPM_tcS2F_2p6_ps_specific/CircuitConfig"
circuit_path = "/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/individuals/O1_P14-13/merged_circuit/ncsThalamocortical_VPM_tcS2F_2p6_ps/CircuitConfig"
proj_recipe_path = "/gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe.xml"

#c = bluepy.Circuit("/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/1x7_0/merged_circuit/ncsThalamocortical_L4_tcS2F_2p0/CircuitConfig")
#c = bluepy.Circuit("/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/ncsThalamocortical_L4_tcS2F_2p0/CircuitConfig")
#c = bluepy.Circuit("/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/ncsThalamocortical_L4_tcS2F_3p0/CircuitConfig")
#c = bluepy.Circuit("/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/merged_circuit/ncsThalamocortical_VPM_tcS2F_3p0_ps/CircuitConfig")

#c = bluepy.Circuit("/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/merged_circuit/ncsThalamocortical_POm_tcS2F_2p6_ps/CircuitConfig")

def plot_syn_density(circuit_path, proj_recipe_path):
    c = bluepy.Circuit(circuit_path)

    idx = c.synapse_spatial_index()

    model = c
    mesoCenter = model.RUN.CONTENTS.CentralHyperColumn

    # column parameters
    # xz coordinates of the hypercolumn center
    columnInfo = model.get_mosaic_geometry()
    tile = columnInfo.get_tile(int(mesoCenter))
    columnCenter = tile.center

    # y (height) of the hypercolumn and layer thicknesses
    recipe = model.recipe 
    ids = numpy.array(map(int,recipe.xpath("//blueColumn/column/layer/@id")))
    thickness = numpy.array(map(float,recipe.xpath("//blueColumn/column/layer/@thickness")))
    shuffle_idx = numpy.argsort(ids)
    layerThickness = thickness[shuffle_idx]
    assert numpy.alltrue(ids[shuffle_idx] == numpy.arange(6)+1)

    # layer tops and bottoms
    tops = numpy.add.accumulate(layerThickness[::-1])[::-1]
    bottoms = tops-layerThickness
    centers = bottoms+layerThickness/2.0


    num_bins = 100
    projSynapses = numpy.zeros(num_bins)
    #top = tops[4-1]
    #bottom = centers[5-1]+200

    top = tops[3-1]
    bottom = bottoms[6-1]

    bin_size = (top-bottom)/num_bins
    xz_size = 150.0
    xz_half_size = xz_size/2


    for bin in range(num_bins):
        # get the segment index
        mytop = top-bin_size*bin
        mybottom = top-bin_size*(bin+1)
        x1 = (columnCenter[0]-xz_half_size, mybottom,columnCenter[1]-xz_half_size)
        x2 = (columnCenter[0]+xz_half_size, mytop,columnCenter[1]+xz_half_size)
        try:
            results = idx.q_window_oncenter(x1, x2)
        except:
            projSynapses[bin]=0
            continue

        #count number of inhibitory synapses
        #projSynapses[layer_idx] = numpy.sum(results[:,11]==0.0)
        projSynapses[bin] = len(results)

    # print inhSynapses
    means = projSynapses/(bin_size*xz_size**2)
    #rel_err = 1/numpy.sqrt(projSynapses)
    #abs_err = means*rel_err
    # print means
    #return means, abs_err


    plt.figure()
    recipe_yprofile_plotter.plot_yprofile(circuit_path, proj_recipe_path)
    #bins = arange(bottom, top, cube_side)
    bins = numpy.linspace(top,bottom-bin_size,num_bins+1)
    #plot(bins[::-1],numpy.hstack((means, means[-1]))[::-1], ls="steps")
    plt.plot(numpy.hstack((means, means[-1]))*1000, bins, 'r:', ls="steps")
    plt.ylabel('y [um]')
    # set background color white
    plt.gcf().patch.set_facecolor('white')
    plt.draw()


if __name__=="__main__":
    
    if len(sys.argv)==3:
        circuit_path = sys.argv[1]
        proj_recipe_path = sys.argv[2]
    else:
        print "Usage: syn_density_plotter.py circuit_path proj_recipe_path"
        sys.exit()



    plot_syn_density(circuit_path, proj_recipe_path)
    plt.show()
