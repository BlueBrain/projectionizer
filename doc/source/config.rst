Configuration file
==================
The configuration file is a yaml file. It can be seen as a dictionary where keys are Luigi tasks and values are parameters specific to each task.

Parameters
----------

**CommonParams** lists params than are common to all tasks.
 - **circuit_config**: the CircuitConfig absolute path.
 - **recipe_path**: The path to the XML recipe that is used by Spykfunc
 - **morphology_path**: The path to the morphology release
 - **sgid_offset**: the offset used for indexing the virtual fibers
 - **n_total_chunks**: in order to prevent RAM from exploding, the computation is splitted into chunks. This is the number of chunks.
 - **oversampling**: the ratio between the number of sampled synapses during the first step and the number of desired synapses. Oversampling is necessary as it allows to remove unwanted synapses with bad connectivity properties while keeping the final number of synapses stable.
 - **layers**: list of tuples with layer names and thicknesses (in um). Thicknesses will be deprecated.
 - **fiber_locations_path**: path to a csv file containing fiber positions and directions (can be generated with `projectionizer generate-fibers[-hex]`)
 - **region**: list of region names to generate projections to
 - **hex_apron_bounding_box**: coordinates of the bounding box of the apron optionally used with columns to reduce edge effects

**SynapseDensity** has a single parameter:
 - **density_params**: is a list where each item describes the synaptic density along a layer (or a portion of it). It is composed of multiple sub-items
  - **density_profile**: a list of 2-tuple (relative position in the item (in %), density unit)
  - **low_layer**: the starting layer name for the item
  - **low_fraction**: the relative position with respect to the start of low_layer
  - **high_layer**: the ending layer name for the item
  - **high_fraction**: the relative position with respect to the start of high_layer

  Example:
    | low_layer: 4
    | low_fraction: 0
    | high_layer: 3
    | high_fraction: 0.5
    | density_profile: [[0.25, 0.01], [0.50, 0.02], [0.75, 0.03]]

    This represents a density profile spanning from the bottom (low_fraction=0) of layer 4 to 50% of the height of layer 3. The first quarter of the span has a density of 0.01, the second as a density of 0.02 and the rest as a density of 0.03.

**FullSample** build the synapse dataframe:
  - **n_slices**: this is a convenience parameter to sample synapses only for a given number of voxels. -1 means all. Other value should **never** be used for scientific purposes.

**FiberAssignment** assigns each sampled synapse to a virtual fiber:
  - **sigma**: The probability of pairing between a fiber and a synapse is proportional to a gaussian of the distance fiber-synapse parameter. This is its sigma.

**ClosestFibersPerVoxel** exists because computing the pairing probabilities between every synapse and every fiber would take forever. It returns a dataframe with the most relevant (ie. closest) fibers for each synapses.
  - **closest_count**: the number of fibers to return for each synapse

**ChooseConnectionsToKeep** is the task responsible for getting rid of 'unbiological' connections; pairs connected by a too small numbers of synapses.
  - **cutoff_var**: Connections are filtered based on there number of synapses. The filter function is a sigmoid function centered at the cutoff value. `cutoff_var` is the width of the sigmoid.

**PruneChunk** prunes out the connections that are not kept
  - **additive_path_distance**: distance to add to the path distance (to make sure sure delay > .1 in simulations)

**WriteSonata** parameterizes the SONATA files (assumes 'sonata' format was used in WriteAll)
  - **target_population**: The name of the target node population (default: All)
  - **mtype**: mtype of the nodes, also used as the target name in the user.target file (default: projections)
  - **node_population**: The name of the created node population (default: projections)
  - **edge_population**: The name of the created edge population (default: projections)
  - **node_file_name**: file name for the sonata node file (default: projections_nodes.h5)
  - **edge_file_name**: file name for the sonata edge file (default: projections_edges.h5)
  - **module_archive**: which archive to load spykfunc and parquet-converters from (default: archive/2021-07)

**VolumeSample** does the spherical sampling for volume transmission projections
  - **radius**: radius (around synapses) to consider for volume transmission (Default: 5um)
  - **additive_path_distance**: distance to add to the path distance (to make sure sure delay > .1 in simulations) (Default: 300um)

**ScaleConductance** scale the conductance of volume transmission projections according to the distance from the synapse
  - **interval**: A tuple giving the linear scale for conductance scaling (Default: [1.0, 0.1])
