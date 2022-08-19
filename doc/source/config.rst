.. _configuration:

Configuration file
==================
The configuration file is a yaml file.
It can be seen as a dictionary where keys are Luigi tasks and values are parameters specific to each task.

For **TL;DR**, see the :ref:`example configuration<Config_ExampleFile>`.

Parameters
----------

.. _Config_CommonParams:

CommonParams
~~~~~~~~~~~~
Lists parameters that are common to all tasks.

.. table::

  ====================== ========= ========= =======================================
  Parameter              Mandatory Default   Description
  ====================== ========= ========= =======================================
  circuit_config         Yes                 The CircuitConfig absolute path.
  physiology_path        Yes                 The path to the XML recipe that is used by Spykfunc
  sgid_offset            Yes                 The offset used for indexing the virtual fibers
  n_total_chunks         Yes                 In order to prevent RAM from exploding, the computation is split into chunks. This is the number of chunks.
  oversampling           Yes                 The ratio between the number of sampled synapses during the first step and the number of desired synapses. Oversampling is necessary as it allows to remove unwanted synapses with bad connectivity properties while keeping the final number of synapses stable.
  layers                 Yes                 List of layer names (as in `hierarchy.json`) arranged from bottom to top (e.g., `[L6, L5, L4, L3, L2, L1]`)
  fiber_locations_path   Yes                 Path to a csv file containing fiber positions and directions. It can be :ref:`generated<Index_CreatingFibers>` with projectionizer.
  regions                No        None      List of region names to generate projections to. If not given, parsed from MANIFEST (should be defined in [common] > [region]). If not defined in manifest, raises an exception.
  hex_apron_bounding_box No        None      Coordinates of the bounding box of the apron optionally used with columns to reduce edge effects (see: :ref:`apron<FAQ_apron>`)
  ====================== ========= ========= =======================================

SynapseDensity
~~~~~~~~~~~~~~
Samples he synapses according to a vertical synapse density profile.

.. table::

  ====================== ========= ========= =======================================
  Parameter              Mandatory Default   Description
  ====================== ========= ========= =======================================
  density_params         Yes                 A list where each item describes the synaptic density along a layer (or a portion of it).
  ====================== ========= ========= =======================================

The `density_params` is composed of multiple sub-items

.. table::

  ====================== ========= ========= =======================================
  Parameter              Mandatory Default   Description
  ====================== ========= ========= =======================================
  density_profile        Yes                 A list of 2-tuple (relative position in the item (in %), density unit)
  low_layer              Yes                 The starting layer name for the item
  low_fraction           Yes                 The relative position with respect to the start of low_layer
  high_layer             Yes                 The ending layer name for the item
  high_fraction          Yes                 The relative position with respect to the start of high_layer
  ====================== ========= ========= =======================================

**Example**

.. code-block:: yaml

    SynapseDensity:
      density_params:         # List of density profile items
        - low_layer: L4       # starts from layer L4
          low_fraction: 0     # starts from bottom of the layer L4
          high_layer: L3      # ends in layer L3
          high_fraction: 0.5  # ends in midway of the layer L3
          density_profile:    # the density profile of the item
            - [0.25, 0.01]    # from 0% to 25% of the span the density is 0.01
            - [0.50, 0.02]    # from 25% to 50%: the density is 0.02
            - [0.75, 0.03]    # from 50 to 75% (and to 100%): the density is 0.03
        - low_layer: L6       # next item start from layer L6
          ...                 # etc.

FiberAssignment
~~~~~~~~~~~~~~~
Assigns each sampled synapse to a virtual fiber

.. table::

  ====================== ========= ========== =======================================
  Parameter              Mandatory Default    Description
  ====================== ========= ========== =======================================
  sigma                  Yes                  The probability of pairing between a fiber and a synapse is proportional to a gaussian of the distance fiber-synapse parameter. This is its sigma.
  ====================== ========= ========== =======================================

ClosestFibersPerVoxel
~~~~~~~~~~~~~~~~~~~~~
Returns a dataframe with the most relevant (ie. closest) fibers for each synapses.
This is done because computing the pairing probabilities between every synapse and every fiber would take forever.

.. table::

  ====================== ========= ========== =======================================
  Parameter              Mandatory Default    Description
  ====================== ========= ========== =======================================
  closest_count          Yes                  The number of fibers to return for each synapse
  ====================== ========= ========== =======================================

ChooseConnectionsToKeep
~~~~~~~~~~~~~~~~~~~~~~~
Is the task responsible for getting rid of 'unbiological' connections; pairs connected by a too small numbers of synapses.

.. table::

  ====================== ========= ========== =======================================
  Parameter              Mandatory Default    Description
  ====================== ========= ========== =======================================
  cutoff_var             Yes                  Connections are filtered based on there number of synapses. The filter function is a sigmoid function centered at the cutoff value. `cutoff_var` is the width of the sigmoid.
  ====================== ========= ========== =======================================

PruneChunk
~~~~~~~~~~
Prunes out the connections that are not kept.

.. table::

  ====================== ========= ========== =======================================
  Parameter              Mandatory Default    Description
  ====================== ========= ========== =======================================
  additive_path_distance No        0.0        Distance to add to the path distance (to make sure sure delay > .1 in simulations)
  ====================== ========= ========== =======================================

WriteSonata
~~~~~~~~~~~
Parameterizes the SONATA files.

.. table::

  ====================== ========= ====================== =======================================
  Parameter              Mandatory Default                Description
  ====================== ========= ====================== =======================================
  mtype                  No        projections            The mtype of the nodes, also used as the target name in the user.target file
  node_population        No        projections            The name of the created node population
  edge_population        No        projections            The name of the created edge population
  node_file_name         No        projections-nodes.h5   File name for the sonata node file
  edge_file_name         No        projections-edges.h5   File name for the sonata edge file
  module_archive         No        archive/2021-07        Which archive to load spykfunc and parquet-converters from
  ====================== ========= ====================== =======================================

VolumeSample
~~~~~~~~~~~~
Does the spherical sampling for volume transmission projections.

.. table::

  ====================== ========= ========== =======================================
  Parameter              Mandatory Default    Description
  ====================== ========= ========== =======================================
  radius                 No        5          radius (around synapses) to consider for volume transmission
  additive_path_distance No        300        distance to add to the path distance (to make sure sure delay > .1 in simulations)
  ====================== ========= ========== =======================================

ScaleConductance
~~~~~~~~~~~~~~~~
Scale the conductance of volume transmission projections according to the distance from the synapse.

.. table::

  ====================== ========= =============== =======================================
  Parameter              Mandatory Default         Description
  ====================== ========= =============== =======================================
  interval               No        [1.0, 0.1]      A tuple giving the linear scale for conductance scaling
  ====================== ========= =============== =======================================

.. _Config_ExampleFile:

Example
-------

.. code-block:: yaml

    ChooseConnectionsToKeep:
      cutoff_var: 1.0
    ClosestFibersPerVoxel:
      closest_count: 25
    CommonParams:
      circuit_config: /gpfs/bbp.cscs.ch/project/proj87/scratch/circuits/SSCX-O1/CircuitConfig
      fiber_locations_path: /gpfs/bbp.cscs.ch/project/proj87/scratch/projections/SSCX-O1/dopamine/dopamine_fibers.csv
      physiology_path: /gpfs/bbp.cscs.ch/project/proj87/scratch/projections/SSCX-O1/dopamine/DA_proj_recipe.xml
      layers:
      - L6
      - L5
      - L4
      - L3
      - L2
      - L1
      n_total_chunks: 1
      oversampling: 1
      regions:
      - mc0_Column
      - mc1_Column
      - mc2_Column
      - mc3_Column
      - mc4_Column
      - mc5_Column
      - mc6_Column
      sgid_offset: 6000000
    FiberAssignment:
      sigma: 50
    PruneChunk:
      additive_path_distance: 300
    SynapseDensity:
      density_params:
      # .
      # .
      # .
      # <truncated for readability>
      # .
      # .
      # .
      - low_layer: L2
        low_fraction: 0.0
        high_layer: L2
        high_fraction: 1.0
        density_profile:
        - - 0.333333333333
          - 0.0004254399737045899
        - - 0.666666666667
          - 0.0004169391749822368
        - - 1.0
          - 0.00041463895885736476
      - low_layer: L1
        low_fraction: 0.0
        high_layer: L1
        high_fraction: 1.0
        density_profile:
        - - 0.294117647059
          - 0.00041463895885736476
        - - 0.588235294118
          - 0.00041463895885736476
    VolumeSample:
      additive_path_distance: 300
      radius: 2
    ScaleConductance:
      interval:
        - 1.0  # conductance = 1.0 * conductance at distance==0
        - 0.1  # conductance = 0.1 * conductance at distance==VolumeSample.radius
