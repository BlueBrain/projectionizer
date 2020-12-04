#!/usr/bin/env python
''' projectionizer setup '''

import imp
from setuptools import setup

VERSION = imp.load_source('projectionizer', "projectionizer/version.py").VERSION

setup(
    name='projectionizer',
    version=VERSION,
    install_requires=[
        'toolz>=0.8',
        'partd>=0.3',
        'luigi>=2.7,<3.0',
        'tables>=3.4',
        'pyarrow>=0.11.1',
        'pyrsistent==0.16.0',  # To fix a bug regarding py27 support of pyrsistent
        # BBP
        'bluepy>=0.10,<=0.14.14',
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
    python_requires='>=2.7,<3',
)
