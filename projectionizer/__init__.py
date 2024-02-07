"""Projectionizer"""

import logging

from projectionizer.version import VERSION as __version__  # pylint: disable=W0611

# Since projectionizer does not build a `multi-index` but merely queries it, momentarily change
# log level to not show the warning of spatial index not having `MPI` support.
logging.getLogger("spatial_index").setLevel(logging.ERROR)

try:
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
finally:
    logging.getLogger("spatial_index").setLevel(logging.ERROR)

# Configure the root logger without touching the LogLevel of it
# Set logging level only for projectionizer. Luigi and `matplotlib` DEBUG logging is too noisy.
# When doing this in `__init__.py`, the Log Level affects all the sub-modules, too
logging.basicConfig()  # Don't set the level of root logger to DEBUG
logging.getLogger(__name__).setLevel(logging.DEBUG)
