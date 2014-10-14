

Density profile 
===============

1) Spatial index the synapses

See for an example:
/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/1x7_0/merged_circuit/ncsThalamocortical_L4_tcS2F

cp proj_nrn.h5 nrn.h5

ln -s ../start.target .
cp ../CircuitConfig .
edit the nrnPath to point to the projection directory.

May need to run this 
https://bbpteam.epfl.ch/svn/user/ebmuller/SynMerge/AddVersionDataset.py
python <path>/AddVersionDataset.py nrn.h5
