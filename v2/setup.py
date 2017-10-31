#!/usr/bin/env python
''' projectionizer setup '''
import os

from setuptools import setup
import projectionizer

VERSION = projectionizer.__version__

############ REQUIREMENTS FINDING
BASEDIR = os.path.dirname(os.path.abspath(__file__))
REQS = []
EXTRA_REQS_PREFIX = 'requirements_'
EXTRA_REQS = {}

import pip
from pip.req import parse_requirements
from optparse import Option


def parse_reqs(reqs_file):
    ''' parse the requirements '''
    options = Option('--workaround')
    options.skip_requirements_regex = None
    # Hack for old pip versions: Versions greater than 1.x
    # have a required parameter "sessions" in parse_requierements
    if pip.__version__.startswith('1.'):
        install_reqs = parse_requirements(reqs_file, options=options)
    else:
        from pip.download import PipSession  # pylint:disable=E0611
        options.isolated_mode = False
        install_reqs = parse_requirements(reqs_file,  # pylint:disable=E1123
                                          options=options,
                                          session=PipSession)
    return [str(ir.req) for ir in install_reqs]

REQS = parse_reqs(os.path.join(BASEDIR, 'requirements.txt'))

#look for extra requirements (ex: requirements_bbp.txt)
for file_name in os.listdir(BASEDIR):
    if not file_name.startswith(EXTRA_REQS_PREFIX):
        continue
    base_name = os.path.basename(file_name)
    (extra, _) = os.path.splitext(base_name)
    extra = extra[len(EXTRA_REQS_PREFIX):]
    EXTRA_REQS[extra] = parse_reqs(file_name)

setup(
    name='projectionizer',
    version=VERSION,
    install_requires=REQS,
    extras_require=EXTRA_REQS,
    packages=['projectionizer',
              ],
    include_package_data=True,
    author='BlueBrain NSE',
    author_email='bbp-ou-nse@groupes.epfl.ch',
    description='Voxel based projections',
    license='BBP-internal-confidential',
    url='http://bluebrain.epfl.ch',
)
