#!/usr/bin/env python
''' projectionizer setup '''

from setuptools import setup

import projectionizer

VERSION = projectionizer.__version__


setup(
    name='projectionizer',
    version=VERSION,
    install_requires=[
        'dask>=0.15',
        'dask[distributed]>=1.16',
        'toolz>=0.8',
        'partd>=0.3',
        'feather-format>=0.4',
        'luigi>=2.7',
        'tables>=3.4',
        # BBP
        'bluepy>=0.10',
        'libFLATIndex>=1.8',
        'voxcell>=2.3',
    ],
    packages=[
        'projectionizer',
    ],
    scripts=[
        'apps/projectionizer',
    ],
    author='BlueBrain NSE',
    author_email='bbp-ou-nse@groupes.epfl.ch',
    description='Voxel based projections',
    license='BBP-internal-confidential',
    url='http://bluebrain.epfl.ch',
)
