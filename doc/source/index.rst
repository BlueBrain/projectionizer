Overview
========

The `projectionizer` aims at sampling synapses with pre-synaptic axons (refered in the code as virtual fibers) coming from outer regions.
It provides the fibers in a sonata format.

.. warning::

    Due to the change of sampling being done by brain-indexer (instead of libFLATIndex), **any of the projection workflow output files created prior to March 2023 should not be used with projectionizer version 3.0.0 or newer** due to the change of the GID indexing.
    For further information, see: :ref:`FAQ<FAQ_Indexing>`.

    If need be, these older output files should be used with one of the projectionizer versions using libFLATIndex (i.e., projectionizer < 3.0.0).

    However, in general, continueing old partially run projectionizer workflows (or removing output files to recreate them) is not recommended and may result in issues or unexpected results.


Morphological constraints
-------------------------

The sampling must be done with respect to the following constraints:
 - the synaptical density along the column
 - the number of synapse per connection (neuron/virtual fiber) density distribution
 - the efferent neuron count distribution (number of neurons a virtual fiber is connected to)

Installation
------------

To install projectionizer, user should create a virtual environment with Python (min version: 3.8), and run:

.. code-block:: text

    pip install pip --upgrade
    pip install projectionizer --index https://bbpteam.epfl.ch/repository/devpi/simple/


Usage
-----

.. note::

    Since synapse parameterization relies on `Spykfunc <https://bbpteam.epfl.ch/documentation/projects/spykfunc/latest/usage.html>`_, Projectionizer needs to be run in an exclusive allocation to ensure that Spykfunc functions as expected.

Starting a new job
~~~~~~~~~~~~~~~~~~

.. note::
    ``create-projections`` and ``volume-transmission`` jobs raise an error if the output directory already exists.

    The job can be resumed or overwritten using ``--resume`` and ``--overwrite`` flags.

**Normal projections**

.. code-block:: text

    projectionizer create-projections -o output_folder -c config_file.yaml

will generate projections using the passed config file, the content of which is described in :ref:`configuration`.

**Volume transmission projections**

.. code-block:: text

    projectionizer volume-transmission -o output_folder -c config_file.yaml

will generate volume transmission projections using the passed config file. Will also run the projections workflow. Implies output in SONATA format.


Resuming a job
~~~~~~~~~~~~~~

.. code-block:: text

    projectionizer [create-projections|volume-transmission] \
        -o output_folder -c config_file.yaml \
        --resume

will resume the job using parameters in the `config.yaml` present in the output folder.
If the contents of the input config file don't match those of the `config.yaml` in the output folder, an error will be raised.

Overwriting a job
~~~~~~~~~~~~~~~~~

.. code-block:: text

    projectionizer [create-projections|volume-transmission] \
        -o output_folder -c config_file.yaml \
        --overwrite

will overwrite (removes the folder and its contents) the job using parameters in the passed `config_file.yaml`.

Dichotomy pipeline
------------------

.. note::
    Dichotomy job raises an error if the output directory already exists. It can't be resumed.

The projection validity is constrained by comparing the connectivity (mean value of the number of synapses per connection) with the experitmental data.
This value is directly influenced by the oversampling value: a lower oversampling will lead to a lower connectivity and vice-versa.

The `dichotomy` subcommand automates the trial-and-error process of finding the correct oversampling value.
It will generate projections with different oversampling values until the given ``connectivity-target`` value is matched.

To select which m-types are considered when measuring connectivity, see the `target_mtypes` field in the :ref:`Config_CommonParams` of the configuration file.

.. code-block:: text

    projectionizer dichotomy -o output_folder -c config_file.yaml \
        --connectivity-target 7.0 \  # the connectivity to reach
        --target-margin 0.2 \        # the accepted tolerance for the connectivity
        --min-param 2.1 \            # the minimum oversampling values
        --max-param 15.0             # the maximum oversampling values


.. _Index_CreatingFibers:

Creating fibers
---------------

.. note::
    Fiber generation jobs raise an error if the output fiber file already exists.

.. code-block:: text

    projectionizer generate-fibers \
        -o output_file.csv \
        -c config.yaml \
        -n 5000

will generate 5000 fibers in region(s) defined in `config.yaml`. Fibers are positioned in the bottom layer (farthest from Pia Mater).
They are acquired by tracing back direction vectors from upper layers.

.. code-block:: text

    projectionizer generate-fibers-hex \
        -o output_file.csv \
        -c config.yaml \
        -n 300 -v 1.0 -y 0.0

command for columns that will generate 300 fibers in region(s) defined in `config.yaml` using K means clustering.
Fibers start from the y-position 0 (`-y 0.0`) towards positive y direction (`-v 1.0`).
In case an :ref:`apron bounding box<Config_CommonParams>` is defined in the config, the fibers are placed inside its boundaries.

.. toctree::
   :hidden:

   self
   theory_practice
   config
   faq
   changelog

Running with SBATCH
-------------------

As with everything, it is often easier to run projectionizer with SBATCH.
There are some things to take into account:

- It needs to be run in exclusive environment: ``--exclusive``
- NVME required: ``--constraint=nvme``
- Use only one task per node: ``--ntasks-per-node=1``

  - Otherwise, `spykfunc` will likely run into issues with multiple schedulers

- Projectionizer does not support multi-node parallelization

  - To be on the safe side, use only one node: ``--nodes=1``
  - Technically, `spykfunc` is perfectly capable to run in multiple nodes but we haven't tested how it behaves when launched from projectionizer

    - if you need more nodes for reason `<insert reason here>`, or just like to live dangerously, you can always give it a shot

- Environment variables that can be tweaked:

  - `PARALLEL_COUNT`

    - the number of CPU cores used for parallelization of certain tasks.

  - `BRAIN_INDEXER_CACHE_SIZE_MB`

    - The cache size (per process) used by brain-indexer.
    - When changing this value, please take into account the number of parallel processes and the overall system memory.

You can use this example to base your sbatch file on:

.. _Index_ExampleSbatch:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=awesome_projections
    #SBATCH --account=proj30
    #SBATCH --nodes=1
    #SBATCH --time=24:00:00  # I hope this isn't going to take that long
    #SBATCH --ntasks-per-node=1
    #SBATCH --constraint=nvme  # spykfunc requires nvme
    #SBATCH --mem=0
    #SBATCH --partition=prod
    #SBATCH --exclusive
    #SBATCH --output=SSCX_proj_%j.log

    # Specify virtual environment, config and output directory
    VENV=./projectionizer_venv/
    CONFIG=./projectionizer_config.yaml
    OUTDIR=./out

    # Number of processes in parallel functions (OOM Error -> use less cores)
    export PARALLEL_COUNT=70

    # Cache size (per process) for brain-indexer, in Mb [default: 4000]
    export BRAIN_INDEXER_CACHE_SIZE_MB=4000

    source $VENV/bin/activate

    projectionizer create-fibers -c $CONFIG -o $OUTDIR
