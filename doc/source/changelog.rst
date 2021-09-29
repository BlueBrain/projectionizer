Changelog
=========

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
- Removed layer thicknesses from config
- Updated documentation
- Parallelization with multiple nodes
- Ensured runs are reproducable
- Renamed `recipe_path` to `physiology_path` in config
- `generate-fibers(-hex)` will now read circuit path, regions and bounding rectangle from the YAML config given as an argument
- Code style improvements
- Improved test coverage and added a restriction to 100% coverage
- Added basic linting and `isort` to tests

Bug Fixes
~~~~~~~~~
- Restrict bluepy<2.3 until next release of MorphIO (see https://github.com/BlueBrain/MorphIO/pull/330)
