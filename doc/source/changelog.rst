.. _changelog:

Changelog
=========

Version v3.0.0
--------------

.. _ver_3_0_0_breaking:

Breaking Changes
~~~~~~~~~~~~~~~~
- continueing tasks started with projectionizer < v3.0.0
- ``python`` 3.7 no longer supported
- as the offset is removed, projections are no longer compatible with BlueConfig

  - ``user.target`` file will no longer be generated

- commandline interface changes:

  - ``projectionizer start`` was changed to ``projectionizer create-projections``
  - ``projectionizer resume`` was removed

    - one should use ``projectionizer create-projections --resume``

  - ``projectionizer <subcommand> --resume`` expects a config file now

    - raises an error if not given or if its contents differ from those of the one in the output folder

- removed values from config (providing these will cause the workflow to crash)

  - ``CommonParams``: ``morphology_path``
  - ``CommonParams``: ``sgid_offset``
  - ``WriteSonata``: ``target_population``

- new config values (not providing these will raise an error)

  - ``CommonParams``: ``segment_index_path``
  - ``CommonParams``: ``projectionizer_version``

- config values moved to ``CommonParams``:

  - ``WriteSonata: module_archive``

- minimum ``module_archive`` version is ``archive/2023-06``

New Features
~~~~~~~~~~~~
- ``python >= 3.8`` support
- ``afferent_section_pos`` is now computed for the synapses

Improvements
~~~~~~~~~~~~
- ``SGID`` will no longer be offset
- Everything uses 0-based GID indexing
- default ``module_archive`` updated to ``archive/2023-06``
- segment indexing is done with ``SpatialIndex`` and ``libFLATIndex`` was removed
- reading more parameters from circuit config instead of the user-provided config file (see :ref:`ver_3_0_0_breaking`)
- using ``click`` to handle the commandline interface


Version v2.0.2
--------------

Improvements
~~~~~~~~~~~~
- Added ``afferent_section_type`` and ``efferent_section_type`` to the resulting ``projections-edges.h5`` file
- Remove version pins from ``bluepy`` and ``bluepysnap`` and require newer versions
- Updated project URLs in setup.py
- Added an :ref:`example configuration file<Config_ExampleFile>`
- Added ``efferent_center_{x,y,z}`` and ``distance_volume_transmission`` fields to volume transmission edges file

Bug Fixes
~~~~~~~~~
- Fixed the workflow for newer (than ``archive/2021-07``) versions of ``spykfunc``
- Restrict ``numpy<1.22`` as the support for ``python3.7`` will be dropped in that release
- Fixed a bug which prevented from running the normal pipeline with a config containing Volume Transmission tasks
- Fixed a bug in Volume Transmission pipeline causing the conductance values to be incorrect
- Changed smallest integer type to ``int16`` for backward compatibility with older ``spykfunc`` versions


Version v2.0.1
--------------

Bug Fixes
~~~~~~~~~
- Restrict ``bluepy==2.2.0`` and ``bluepysnap==0.11.0`` due to inefficient dependency resolving in pip


Version v2.0.0
--------------

New Features
~~~~~~~~~~~~
- Start of Changelog
- SONATA format (nrn, syn2 no longer supported)
- Spykfunc Parameterization
- Fiber generation
- Dropped python v2.7 support

Improvements
~~~~~~~~~~~~
- Updated documentation contents and appearance
- Ensured runs are reproducible
- Introduced changes in :ref:`configuration`

  - added sections ``WriteSonata``, ``VolumeSample``, ``ScaleConductance``
  - removed section ``WriteNrn``, ``WriteSyn2``
  - changed parameters in section ``CommonParams``

    - added ``hex_apron_bounding_box``, ``morphology_path``
    - renamed ``recipe_path`` to ``physiology_path``
    - renamed ``hex_fiber_locations`` to ``fiber_locations_path``
    - removed layer thicknesses from ``layers``
    - removed ``geometry`` and ``voxel_path``

  - removed ``n-slices`` from ``FullSample``
  - removed ``target-name`` from ``WriteUserTargetTxt``

    - target name is now automatically the same as the ``mtype`` in ``WriteSonata``

- ``generate-fibers(-hex)`` will now read circuit path, regions and bounding rectangle from the YAML config given as an argument
- Code style improvements
- Improved test coverage and added a restriction to 100% coverage
- Added basic linting and ``isort`` to tests

Bug Fixes
~~~~~~~~~
- Restrict bluepy<2.3 until next release of MorphIO (see https://github.com/BlueBrain/MorphIO/pull/330)
