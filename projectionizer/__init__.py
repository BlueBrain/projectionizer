'''docstring'''
from projectionizer.version import VERSION as __version__  # pylint: disable=W0611

#  Luigi needs this to be able to call all the commands from the CLI
from projectionizer.step_0_sample import (VoxelSynapseCount, Height, FullSample,
                                          SampleChunk, SynapseDensity)
from projectionizer.step_1_assign import (VirtualFibersNoOffset, ClosestFibersPerVoxel,
                                          SynapseIndices, CandidateFibersPerSynapse, Centroids,
                                          SynapticDistributionPerAxon, FiberAssignment)
from projectionizer.step_2_prune import (GroupByConnection, ReduceGroupByConnection, CutoffMeans,
                                         ChooseConnectionsToKeep, PruneChunk, ReducePrune)
from projectionizer.step_3_write import (WriteSummary, WriteNrnH5, WriteUserTargetTxt,
                                         VirtualFibers, SynapseCountPerConnectionL4PC, WriteAll)
