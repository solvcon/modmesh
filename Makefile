# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

# Build modmesh Python extension (even when the timestamp is clean):
#   make
# Build verbosely:
#   make VERBOSE=1
# Build with clang-tidy
#   make USE_CLANG_TIDY=ON

SETUP_FILE ?= ./setup.mk

ifneq (,$(wildcard $(SETUP_FILE)))
	include $(SETUP_FILE)
endif

# To workaround macos SIP: https://github.com/solvcon/modmesh/pull/16.
# Additional configuration can be loaded from SETUP_FILE.
RUNENV += PYTHONPATH=$(MODMESH_ROOT)

SKIP_PYTHON_EXECUTABLE ?= OFF
HIDE_SYMBOL ?= OFF
DEBUG_SYMBOL ?= ON
MODMESH_PROFILE ?= OFF
USE_CLANG_TIDY ?= OFF
CMAKE_BUILD_TYPE ?= Release
MODMESH_ROOT ?= $(shell pwd)
CMAKE_INSTALL_PREFIX ?= $(MODMESH_ROOT)/build/fakeinstall
CMAKE_LIBRARY_OUTPUT_DIRECTORY ?= $(MODMESH_ROOT)/modmesh
CMAKE_ARGS ?=
VERBOSE ?=

pyextsuffix := $(shell python3-config --extension-suffix)
pyvminor := $(shell python3 -c 'import sys; print("%d%d" % sys.version_info[0:2])')

ifeq ($(CMAKE_BUILD_TYPE), Debug)
	BUILD_PATH ?= build/dbg$(pyvminor)
else
	BUILD_PATH ?= build/dev$(pyvminor)
endif

PYTEST ?= $(shell which py.test-3)
ifeq ($(PYTEST),)
	PYTEST := $(shell which pytest)
endif
ifneq ($(VERBOSE),)
	PYTEST_OPTS ?= -v -s
else
	PYTEST_OPTS ?=
endif

.PHONY: default
default: buildext

.PHONY: clean
clean:
	rm -f $(MODMESH_ROOT)/modmesh/_modmesh$(pyextsuffix)
	make -C $(BUILD_PATH) clean

.PHONY: cmakeclean
cmakeclean:
	rm -f $(MODMESH_ROOT)/modmesh/_modmesh$(pyextsuffix)
	rm -rf $(BUILD_PATH)

.PHONY: pytest
pytest: $(MODMESH_ROOT)/modmesh/_modmesh$(pyextsuffix)
	env $(RUNENV) \
		$(PYTEST) $(PYTEST_OPTS) tests/

.PHONY: flake8
flake8:
	make -C $(BUILD_PATH) VERBOSE=$(VERBOSE) flake8

.PHONY: cmake
cmake: $(BUILD_PATH)/Makefile

.PHONY: buildext
buildext: $(MODMESH_ROOT)/modmesh/_modmesh$(pyextsuffix)

.PHONY: install
install: cmake
	make -C $(BUILD_PATH) VERBOSE=$(VERBOSE) install

$(MODMESH_ROOT)/modmesh/_modmesh$(pyextsuffix): $(BUILD_PATH)/Makefile
	make -C $(BUILD_PATH) VERBOSE=$(VERBOSE) _modmesh_py
	touch $@

$(BUILD_PATH)/Makefile: CMakeLists.txt Makefile
	mkdir -p $(BUILD_PATH) ; \
	cd $(BUILD_PATH) ; \
	env $(RUNENV) \
		cmake $(MODMESH_ROOT) \
		-DCMAKE_INSTALL_PREFIX=$(CMAKE_INSTALL_PREFIX) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CMAKE_LIBRARY_OUTPUT_DIRECTORY) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DSKIP_PYTHON_EXECUTABLE=$(SKIP_PYTHON_EXECUTABLE) \
		-DHIDE_SYMBOL=$(HIDE_SYMBOL) \
		-DDEBUG_SYMBOL=$(DEBUG_SYMBOL) \
		-DUSE_CLANG_TIDY=$(USE_CLANG_TIDY) \
		-DLINT_AS_ERRORS=ON \
		-DMODMESH_PROFILE=$(MODMESH_PROFILE) \
		$(CMAKE_ARGS)
