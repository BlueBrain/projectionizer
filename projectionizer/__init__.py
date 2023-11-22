"""Projectionizer"""
import logging

#  Luigi needs this to be able to call all the commands from the `CLI`
from projectionizer.step_0_sample import (
    FullSample,
    Height,
    SampleChunk,
    SynapseDensity,
    VoxelSynapseCount,
)
from projectionizer.step_1_assign import (
    CandidateFibersPerSynapse,
    ClosestFibersPerVoxel,
    FiberAssignment,
    SynapseIndices,
    VirtualFibers,
)
from projectionizer.step_2_prune import (
    ChooseConnectionsToKeep,
    CutoffMeans,
    GroupByConnection,
    PruneChunk,
    ReduceGroupByConnection,
    ReducePrune,
)
from projectionizer.step_3_write import RunAll, SynapseCountPerConnectionTarget
from projectionizer.version import VERSION as __version__  # pylint: disable=W0611

# Configure the root logger without touching the LogLevel of it
# Set logging level only for projectionizer. Luigi and `matplotlib` DEBUG logging is too noisy.
# When doing this in `__init__.py`, the Log Level affects all the sub-modules, too
logging.basicConfig()  # Don't set the level of root logger to DEBUG
logging.getLogger(__name__).setLevel(logging.DEBUG)
