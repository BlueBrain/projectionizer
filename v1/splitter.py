import sys
import os
# add path to where this .py lives
sys.path.append(os.path.split(os.path.abspath(__file__))[0])

#Finally, split the result into many files:
from tools.split_synapse_files import split_nrn
split_nrn('proj_nrn.h5',8192)
