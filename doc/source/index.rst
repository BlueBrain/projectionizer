Overview
========

The `projectionizer` aims at sampling synapses with pre-synaptic axons (refered in the code as virtual fibers) coming from outer regions.
It provides the fibers in a sonata format.

Morphological constraints
-------------------------

The sampling must be done with respect to the following constraints:
 - the synaptical density along the column
 - the number of synapse per connection (neuron/virtual fiber) density distribution
 - the efferent neuron count distribution (number of neurons a virtual fiber is connected to)

Installation
------------

Since it depends on FLATIndex, it cannot be installed through `spack` currently, and should create a virtual environment with Python 3.7, and run:

.. code-block:: bash

    pip install pip --upgrade`
    pip install projectionizer --index https://bbpteam.epfl.ch/repository/devpi/simple/


Usage
-----

.. note::

    Since sonata format relies on `Spykfunc <https://bbpteam.epfl.ch/documentation/projects/spykfunc/latest/usage.html>`_ (if output_type is sonata), Projectionizer needs to be run in an exclusive allocation to ensure that Spykfunc functions as expected.

Starting a new job
~~~~~~~~~~~~~~~~~~

**Normal projections**

.. code-block:: bash

    projectionizer start -o output_folder -c config_file.yaml

will generate projections using the passed config file, the content of which is described in :ref:`configuration`.

**Volume transmission projections**

.. code-block:: bash

    projectionizer volume-transmission -o output_folder -c config_file.yaml

will generate volume transmission projections using the passed config file. Will also run the projections workflow. Implies output in SONATA format.

.. note::
    The job won't start automatically if a `config.yaml` file is already present in the output directory. It give an error and hinting that you should use `--resume` or `--overwrite` flag.

Resuming a job
~~~~~~~~~~~~~~

.. code-block:: bash

    projectionizer resume -o output_folder

and

.. code-block:: bash

    projectionizer [start|volume-transmission] -o output_folder --resume

will resume the job using parameters in the `config.yaml` present in the folder.

Overwriting a job
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    projectionizer [start|volume-transmission] -o output_folder -c config_file.yaml --overwrite

will overwrite (removes the folder and its contents) the job using parameters in the passed `config_file.yaml`.

..  Dichotomy pipeline
    ------------------

    The projection validity is constrained by comparing the L4PC connectivity (mean value of the number of synapses per connection in Layer 4 Pyramidal Cells (L4PC)) with the experitmental data of `~7.0`. This value is directly influenced by the oversampling value: a lower oversampling will lead to a lower connectivity and vice-versa.

    The `dichotomy` sub command automates the trial-and-error process of finding the correct oversampling value. It will generate projections with different oversampling values until the experimental L4PC connectivity value is matched.

    .. code-block:: bash

        projectionizer dichotomy -o . -c config_file.yaml --connectivity-target 7.0 --min-param 2.1 --max-param 15.0 --target-margin 0.2

    can be used to launch the dichotomy.

    - connectivity-target is the L4PC connectivity to reach
    - target-margin is the accepted tolerance for the L4PC connectivity
    - min-param is the minimum oversampling values
    - max-param is the maximum oversampling values

Creating fibers
---------------

.. code-block:: bash

    projectionizer generate-fibers \
        -o output_file.csv \
        -c config.yaml \
        -n 5000

will generate 5000 fibers in region(s) defined in `config.yaml`. Fibers are positioned in the bottom layer (farthest from Pia Mater).
They are acquired by tracing back direction vectors from upper layers.

.. code-block:: bash

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
