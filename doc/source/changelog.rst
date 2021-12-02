Changelog
=========

Version v2.0.2
--------------

Improvements
~~~~~~~~~~~~
- Remove version pins from `bluepy` and `bluepysnap` and require newer versions
- Updated project URLs in setup.py

Bug Fixes
~~~~~~~~~
- Restrict `numpy<1.22` as the support for `python3.7` will be dropped in that release


Version v2.0.1
--------------

Bug Fixes
~~~~~~~~~
- Restrict `bluepy==2.2.0` and `bluepysnap==0.11.0` due to inefficient dependency resolving in pip


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

  - added sections `WriteSonata`, `VolumeSample`, `ScaleConductance`
  - removed section `WriteNrn`, `WriteSyn2`
  - changed parameters in section `CommonParams`

    - added `hex_apron_bounding_box`, `morphology_path`
    - renamed `recipe_path` to `physiology_path`
    - renamed `hex_fiber_locations` to `fiber_locations_path`
    - removed layer thicknesses from `layers`
    - removed `geometry`' `voxel_path`

- `generate-fibers(-hex)` will now read circuit path, regions and bounding rectangle from the YAML config given as an argument
- Code style improvements
- Improved test coverage and added a restriction to 100% coverage
- Added basic linting and `isort` to tests

Bug Fixes
~~~~~~~~~
- Restrict bluepy<2.3 until next release of MorphIO (see https://github.com/BlueBrain/MorphIO/pull/330)
