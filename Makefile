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
HIDE_SYMBOL ?= ON
DEBUG_SYMBOL ?= ON
MODMESH_PROFILE ?= OFF
BUILD_METAL ?= OFF
BUILD_QT ?= ON
USE_CLANG_TIDY ?= OFF
CMAKE_BUILD_TYPE ?= Release
MAKE_PARALLEL ?= -j
MODMESH_ROOT ?= $(shell pwd)
CMAKE_INSTALL_PREFIX ?= $(MODMESH_ROOT)/build/fakeinstall
CMAKE_LIBRARY_OUTPUT_DIRECTORY ?= $(MODMESH_ROOT)/modmesh
# Use CMAKE_PREFIX_PATH to make it easier to build with Qt, e.g.,
# CMAKE_PREFIX_PATH=/path/to/qt/6.2.3/macos
CMAKE_PREFIX_PATH ?=
CMAKE_ARGS ?=
VERBOSE ?=
FORCE_CLANG_FORMAT ?=
QT3D_USE_RHI ?= OFF

# !!! NOTE: USING ANY VENV IS STRONGLY DISCOURAGED IN DEVELOPING MODMESH !!!
# This treatment is a "smarter" way to find python3-config executable.
# In case Python is not system Python. For example. Python virtual environment
# is used.
# However, please note a Python virtual environment is strongly discouraged in
# developing modmesh. We do not actively resolve bugs related to any virtual
# env including venv or conda.
# See https://github.com/solvcon/modmesh/pull/177 for more details.
WHICH_PYTHON := $(shell which python3)
REALPATH_PYTHON := $(realpath $(WHICH_PYTHON))
export DIRNAME_PYTHON := $(dir $(REALPATH_PYTHON))

pyextsuffix := $(shell $(DIRNAME_PYTHON)/python3-config --extension-suffix)
pyvminor := $(shell python3 -c 'import sys; print("%d%d" % sys.version_info[0:2])')

ifeq ($(CMAKE_BUILD_TYPE), Debug)
	BUILD_PATH ?= build/dbg$(pyvminor)
else
	BUILD_PATH ?= build/dev$(pyvminor)
endif
export BUILD_PATH

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

.PHONY: cmake
cmake: $(BUILD_PATH)/Makefile

.PHONY: xcode
xcode: $(BUILD_PATH)_xcode/Makefile

CMAKE_CMD = cmake $(MODMESH_ROOT) \
	-DCMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH) \
	-DCMAKE_INSTALL_PREFIX=$(CMAKE_INSTALL_PREFIX) \
	-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CMAKE_LIBRARY_OUTPUT_DIRECTORY) \
	-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
	-DSKIP_PYTHON_EXECUTABLE=$(SKIP_PYTHON_EXECUTABLE) \
	-DHIDE_SYMBOL=$(HIDE_SYMBOL) \
	-DDEBUG_SYMBOL=$(DEBUG_SYMBOL) \
	-DBUILD_METAL=$(BUILD_METAL) \
	-DBUILD_QT=$(BUILD_QT) \
	-DUSE_CLANG_TIDY=$(USE_CLANG_TIDY) \
	-DLINT_AS_ERRORS=ON \
	-DMODMESH_PROFILE=$(MODMESH_PROFILE) \
	-DQT3D_USE_RHI=$(QT3D_USE_RHI) \
	$(CMAKE_ARGS)

$(BUILD_PATH)/Makefile: CMakeLists.txt Makefile
	mkdir -p $(BUILD_PATH) ; \
	cd $(BUILD_PATH) ; \
	env $(RUNENV) $(CMAKE_CMD)

$(BUILD_PATH)_xcode/Makefile: CMakeLists.txt Makefile
	mkdir -p $(BUILD_PATH)_xcode ; \
	cd $(BUILD_PATH)_xcode ; \
	env $(RUNENV) $(CMAKE_CMD) -G Xcode

.PHONY: buildext
buildext: cmake
	cmake --build $(BUILD_PATH) --target _modmesh_py VERBOSE=$(VERBOSE) $(MAKE_PARALLEL)

.PHONY: install
install: cmake
	cmake --build $(BUILD_PATH) --target $@ VERBOSE=$(VERBOSE) $(MAKE_PARALLEL)

.PHONY: pytest
pytest: buildext
	env $(RUNENV) \
		$(PYTEST) $(PYTEST_OPTS) tests/

PROFFILES = $(shell find profiling -type f -name 'profile_*.py' | sort)
PROFRESDIR = profiling/results

.PHONY: pyprof
pyprof: buildext $(PROFFILES)
	@mkdir -p profiling/results
	@mkdir -p profiling/results/png
	@for fn in $(PROFFILES); \
	do \
		outfn=$${fn%%.py}; \
		outfn=profiling/results/$${outfn##profiling/}.output; \
		echo "$(WHICH_PYTHON) $${fn} > $${outfn}"; \
		env $(RUNENV) \
			$(WHICH_PYTHON) $${fn} > $${outfn} || exit 1; \
	done

.PHONY: pilot
pilot: cmake
	cmake --build $(BUILD_PATH) --target $@ VERBOSE=$(VERBOSE) $(MAKE_PARALLEL)

.PHONY: gtest
gtest: cmake
	cmake --build $(BUILD_PATH) --target run_gtest VERBOSE=$(VERBOSE) $(MAKE_PARALLEL)

.PHONY: run_pilot_pytest
run_pilot_pytest: pilot
	cmake --build $(BUILD_PATH) --target $@ VERBOSE=$(VERBOSE)

.PHONY: standalone_buffer_setup
standalone_buffer_setup:
	$(MAKE) -C contrib/standalone_buffer copy

.PHONY: standalone_buffer
standalone_buffer:
	$(MAKE) -C contrib/standalone_buffer build
	$(MAKE) -C contrib/standalone_buffer run

CFFILES = $(shell find cpp gtests -type f -name '*.[ch]pp' | sort)
ifeq ($(CFCMD),)
	ifeq ($(FORCE_CLANG_FORMAT),)
		CFCMD = clang-format --dry-run
	else ifeq ($(FORCE_CLANG_FORMAT),inplace)
		CFCMD = clang-format -i
	else
		CFCMD = clang-format --dry-run -Werror
	endif
endif

.PHONY: cformat
cformat: $(CFFILES)
	@for fn in $(CFFILES) ; \
	do \
		echo "$(CFCMD) $${fn}:"; \
		$(CFCMD) $${fn} ; ret=$$? ; \
		if [ $${ret} -ne 0 ] ; then exit $${ret} ; fi ; \
	done

.PHONY: cinclude
cinclude: $(CFFILES)
	if [ "$(shell ag \#include\ \*\" cpp/)" != "" ] ; then exit 1 ; fi

.PHONY: flake8
flake8:
	cmake --build $(BUILD_PATH) --target $@

.PHONY: checkascii
checkascii:
	$(WHICH_PYTHON) contrib/lint/check_ascii.py

.PHONY: checktws
checktws:
	$(WHICH_PYTHON) contrib/lint/check_ascii.py --check-tws

.PHONY: lint
lint: cformat cinclude flake8 checkascii checktws

.PHONY: clean
clean:
	rm -f $(MODMESH_ROOT)/modmesh/_modmesh$(pyextsuffix)
	rm -f $(MODMESH_ROOT)/_modmesh$(pyextsuffix)
	make -C $(BUILD_PATH) clean

.PHONY: cmakeclean
cmakeclean:
	rm -f $(MODMESH_ROOT)/modmesh/_modmesh$(pyextsuffix)
	rm -f $(MODMESH_ROOT)/_modmesh$(pyextsuffix)
	rm -rf $(BUILD_PATH)
