#!/usr/bin/env python
""" projectionizer setup """

import imp

from setuptools import setup

VERSION = imp.load_source("projectionizer", "projectionizer/version.py").VERSION

setup(
    name="projectionizer",
    version=VERSION,
    install_requires=[
        "toolz>=0.8",
        "partd>=0.3",
        "luigi>3.0",
        "tables>=3.4",
        "pyarrow>=0.11.1",
        # BBP
        # until next release of morphio, restict both bluepy and bluepysnap to specific versions.
        # See: https://github.com/BlueBrain/MorphIO/pull/330 and the discussion in
        #      https://bbpgitlab.epfl.ch/nse/projectionizer/-/merge_requests/15
        "bluepy==2.2.0",
        "bluepysnap==0.11.0",
        "libFLATIndex>=1.8.11",
        "voxcell>=3",
    ],
    extras_require={"docs": ["sphinx", "sphinx-bluebrain-theme"]},
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
        "Source": "https://bbpcode.epfl.ch/code/#/admin/projects/building/Projectionizer",
    },
    packages=[
        "projectionizer",
    ],
    scripts=[
        "apps/projectionizer",
    ],
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    description="Voxel based projections",
    license="BBP-internal-confidential",
    url="http://bluebrain.epfl.ch",
    include_package_data=True,
    python_requires=">=3.7,<3.8",  # no libFLATIndex available for py38
)
