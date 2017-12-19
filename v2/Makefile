#modules that have tests
TEST_MODULES=projectionizer

#modules that are installable (ie: ones w/ setup.py)
INSTALL_MODULES=.

# Ignore directories for pep8 and pylint (on top of tests and doc)
IGNORE_LINT=examples

#packages to cover
COVER_PACKAGES=projectionizer
#documentation to build, separated by spaces
DOC_MODULES=

PYTHON_PIP_VERSION="pip==9.0.1"

#NOSEOPS=--exclude-dir=bluepy/bluepy/v2/experimental --exclude-dir=bluepy/bluepy/geometry --exclude-dir=bluepy/bluepy/serializers
#OPTIONAL_FEATURES:='[bbp,nexus,extension_tests]'

##### DO NOT MODIFY BELOW #####################

CI_REPO?=ssh://bbpcode.epfl.ch/platform/ContinuousIntegration.git
CI_DIR?=ContinuousIntegration
CI_REQS=requirements_dev.txt
FETCH_CI := $(shell \
        if [ ! -d $(CI_DIR) ]; then \
            git clone $(CI_REPO) $(CI_DIR) > /dev/null ;\
        fi;\
        echo $(CI_DIR) )
include $(FETCH_CI)/python/common_makefile
