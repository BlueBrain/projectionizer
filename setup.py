#!/usr/bin/env python
''' projectionizer setup '''

import imp
from setuptools import setup

import sys
if sys.version_info[0] == 3:
    sys.exit('Sorry, Python 3.x is not supported')

VERSION = imp.load_source('projectionizer', "projectionizer/version.py").VERSION

setup(
    name='projectionizer',
    version=VERSION,
    install_requires=[
        'dask[distributed]>=0.17',
        'toolz>=0.8',
        'partd>=0.3',
        'luigi>=2.7',
        'tables>=3.4',
        'pyarrow>=0.11.1',
        # BBP
        'bluepy>=0.10',
        'libFLATIndex>=1.8.11',
        'voxcell>=2.5',
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
    include_package_data=True,
)
