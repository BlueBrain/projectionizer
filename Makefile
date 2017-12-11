#modules that have tests
TEST_MODULES=v2/projectionizer

#modules that are installable (ie: ones w/ setup.py)
INSTALL_MODULES=v2

#packages to cover
COVER_PACKAGES=v2/projectionizer

#documentation to build
DOC_MODULES=

IGNORE_LINT=v1

##### DO NOT MODIFY BELOW #####################

CI_REPO?=ssh://bbpcode.epfl.ch/platform/ContinuousIntegration.git
CI_DIR?=ContinuousIntegration

FETCH_CI := $(shell \
		if [ ! -d $(CI_DIR) ]; then \
			git clone $(CI_REPO) $(CI_DIR) > /dev/null ;\
		fi;\
		echo $(CI_DIR) )
include $(FETCH_CI)/python/common_makefile
