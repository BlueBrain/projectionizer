#!/usr/bin/env python
""" projectionizer setup """

import importlib.util

from setuptools import setup

spec = importlib.util.spec_from_file_location("projectionizer.version", "projectionizer/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

setup(
    name="projectionizer",
    version=VERSION,
    install_requires=[
        "bluepysnap>=3.0.1,<4.0.0",
        "click>=8.0,<9.0",
        "h5py<4.0.0",
        "importlib_resources>=5.0.0",
        "libsonata<1.0.0",
        "luigi>3.0",
        "matplotlib<4.0.0",
        "morphio<4.0.0",
        "numpy>=1.19",
        "pandas<2.0.0",
        "pyarrow>=0.11.1",
        "pyyaml<7.0",
        "scipy<2.0.0",
        "brain-indexer>=3.0,<4.0",
        "tqdm<5.0.0",
        "voxcell>=3",
    ],
    extras_require={"docs": ["sphinx", "sphinx-bluebrain-theme"]},
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
        "Source": "https://bbpgitlab.epfl.ch/nse/projectionizer",
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
    url="https://bbpteam.epfl.ch/documentation/projects/projectionizer/latest/",
    include_package_data=True,
    python_requires=">=3.9",
)
