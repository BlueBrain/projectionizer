'''docstring'''
import logging

from projectionizer.version import VERSION as __version__  # pylint: disable=W0611

#  Luigi needs this to be able to call all the commands from the CLI
from projectionizer.step_0_sample import (VoxelSynapseCount, Height, FullSample,
                                          SampleChunk, SynapseDensity)
from projectionizer.step_1_assign import (VirtualFibersNoOffset, ClosestFibersPerVoxel,
                                          SynapseIndices, CandidateFibersPerSynapse,
                                          FiberAssignment)
from projectionizer.step_2_prune import (GroupByConnection, ReduceGroupByConnection, CutoffMeans,
                                         ChooseConnectionsToKeep, PruneChunk, ReducePrune)
from projectionizer.step_3_write import (WriteUserTargetTxt, VirtualFibers,
                                         SynapseCountPerConnectionTarget, WriteAll)

# Configure the root logger without touching the LogLevel of it
# Set logging level only for projectionizer. Luigi and matplotlib DEBUG logging is too noisy.
# When doing this in init, the Log Level affects all the submodules, too
logging.basicConfig()  # Don't set the level of root logger to DEBUG
logging.getLogger(__name__).setLevel(logging.DEBUG)
