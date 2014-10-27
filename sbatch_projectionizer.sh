#!/bin/sh
#SBATCH -J prjctnzr
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --partition=prod
#SBATCH --output=prjctnzr.log
#SBATCH --error=prjctnzr.err
#SBATCH --account=proj1

export PYTHONPATH=/gpfs/bbp.cscs.ch/project/proj1/software/legacy-spatialindexer/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/bbp.cscs.ch/project/proj1/software/legacy-spatialindexer/lib

. ~/venv/bin/activate

srun bash -c ". ~/venv/bin/activate && python projection_runner_tcs2f.py"


