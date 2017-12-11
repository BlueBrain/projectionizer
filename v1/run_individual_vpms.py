import os
import bluepy

def generate_vpm(cfg_file, xml_file):
    circuit_dir = os.path.split(cfg_file)[0]
    out_dir = os.path.join(circuit_dir, "ncsThalamocortical_VPM_os2p6")
    out_dir_ps = os.path.join(circuit_dir, "ncsThalamocortical_VPM_tcS2F_2p6_ps")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if not os.path.exists(out_dir_ps):
        os.mkdir(out_dir_ps)

    ret = os.system("""
    . ~/venv/bin/activate
    export PYTHONPATH=/gpfs/bbp.cscs.ch/project/proj1/software/legacy-spatialindexer/lib:$PYTHONPATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/bbp.cscs.ch/project/proj1/software/legacy-spatialindexer/lib
    cd %s
    srun --account=proj1 -t 3-00:00:00 --mem=28000 -n1 --partition=prod --job-name="prjctnzr" bash -c ". ~/venv/bin/activate && python $HOME/src/Projectionizer-git/projection_runner_tcs2f.py %s %s %s 2>&1 | tee prjctnzr.log"
    """ % (out_dir, cfg_file, xml_file, out_dir))

    if ret!=0:
        print "projection_runner_tcs2f.py failed."
        sys.exit(1)

    ret = os.system("""
    cd %s
    . ~/venv/bin/activate
    python ~/src/bbp-user-ebmuller/SynMerge/TCSynMerge.py
    python ~/src/bbp-user-ebmuller/experiments/thalamocortical_projection/plots/transpose_proj_nrn.py
    """ % (out_dir,))

    if ret!=0:
        print "merge and transpose failed."
        sys.exit(1)


    ret = os.system("""
    srun --account=proj1 --mem=28000 -n1 -t 3-00:00:00 --partition=prod --job-name="prune_runner_ps" bash -c ". ~/venv/bin/activate && cd %s && python $HOME/src/Projectionizer-git/projection_prune_runner_ps.py %s 2>&1 | tee prune_runner_ps.log"
    """ % (out_dir, cfg_file))

    if ret!=0:
        print "prune_runner_ps failed."
        sys.exit(1)


    ret = os.system("""
    cp %s/proj_nrn_efferent_s2f.h5 %s/proj_nrn_efferent.h5
    cd %s
    ln -s proj_nrn_efferent.h5 proj_nrn_efferent.h5.0
    python $HOME/src/bbp-user-ebmuller/experiments/thalamocortical_projection/plots/transpose_proj_nrn_efferent.py
    """ % (out_dir, out_dir_ps, out_dir_ps) )

    if ret!=0:
        print "transpose failed."
        sys.exit(1)

    ret = os.system("""
    cd %s
    python $HOME/src/Projectionizer-git/splitter.py
    """ % (out_dir_ps) )

    if ret!=0:
        print "splitter failed."
        sys.exit(1)
          
    # synapse spatial index

    # re-write CircuitConfig with alternate nrnPath pointing to VPM synapses dir
    c = bluepy.Circuit(cfg_file)
    t = c.config.entry_map["Default"]
    t.CONTENTS.nrnPath = os.path.join(os.path.split(t.CONTENTS.nrnPath)[0], "ncsThalamocortical_VPM_tcS2F_2p6_ps")
    with file(os.path.join(out_dir_ps, "CircuitConfig"), "w") as f:
        c.config.write(f)

    # generate "build_synapse_index.py"
    build_index_template = """

import os
import sys
import libFLATIndex as FLATIndex
print FLATIndex
from glob import glob

print "Running synapse index ..."
FLATIndex.buildIndex('%s/CircuitConfig',"synapse","SYNAPSE",0,"Mosaic")

print "Making index files read-only"
idx_files = glob("SYNAPSE"+"_*")
for file in idx_files:
    print file
    os.chmod(file,0444)

    """ % (out_dir_ps)

    with file(os.path.join(out_dir_ps, "build_synapse_index.py"), "w") as f:
        print >>f, build_index_template

    # minor operations
    os.system("""
    cd %s
    ln -s ../start.target .
    ln -s proj_nrn.h5 nrn.h5
    """ % (out_dir_ps))

    #run spatial indexer
    ret = os.system("""
    srun --account=proj1 -t 3-00:00:00 --mem=28000 -n1 --partition=prod --job-name="vpm-indexer" bash -c "cd %s && export PYTHONPATH=/gpfs/bbp.cscs.ch/project/proj1/software/legacy-spatialindexer/lib:$PYTHONPATH && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/bbp.cscs.ch/project/proj1/software/legacy-spatialindexer/lib && python build_synapse_index.py 2>&1 | tee indexer.log"
    """ % out_dir_ps)

    if ret!=0:
        print "indexer failed."
        sys.exit(1)

    
if __name__=="__main__":
    import sys
    if len(sys.argv)==3:
        cfg_file = sys.argv[1]
        xml_file = sys.argv[2]
    else:
        print "Usage: run_individual_vpms.py cfg_file  xml_file"
        sys.exit(0)

    generate_vpm(cfg_file, xml_file)

# cd $HOME/src/Projectionizer-git/

# python run_individual_vpms.py /gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/individuals/O1_P14-13/merged_circuit/CircuitConfig /gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6.xml

# python run_individual_vpms.py /gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/individuals/O1_P14-14/merged_circuit/CircuitConfig /gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6.xml

# python run_individual_vpms.py /gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/individuals/O1_P14-15/merged_circuit/CircuitConfig /gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6.xml

# python run_individual_vpms.py /gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/individuals/O1_P14-16/merged_circuit/CircuitConfig /gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6.xml


# python run_individual_vpms.py /gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/individuals/O1_P14-17/merged_circuit/CircuitConfig /gpfs/bbp.cscs.ch/home/ebmuller/src/bbp-svn-recipe/Projection_Recipes/Thalamocortical_VPM/thalamocorticalProjectionRecipe_O1_TCs2f_7synsPerConn_os2p6.xml



