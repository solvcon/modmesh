# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

MMROOT := $(realpath ../..)
SETUPROOT ?= $(MMROOT)/$(BUILD_PATH)/standalone_buffer

ifeq ($(SETUPROOT),)
	SETUPROOT := local_standalone_buffer
endif

CFLAGS := \
	-std=c++17 -fPIC \
	-I$(SETUPROOT) \
	-I$(shell python3 -c 'import numpy as np; print(np.get_include())') \
	-I$(shell $(DIRNAME_PYTHON)/python3-config --prefix)/include \
	$(shell $(DIRNAME_PYTHON)/python3-config --cflags)

LDFLAGS := \
	-shared \
	-std=c++17 \
	-L$(shell $(DIRNAME_PYTHON)/python3-config --prefix)/lib \
	-lpython$(shell python3 -c 'import sys; print("%d.%d" % sys.version_info[0:2])') \
	$(shell $(DIRNAME_PYTHON)/python3-config --ldflags) \

MODPATH := $(SETUPROOT)/modbuf$(shell $(DIRNAME_PYTHON)/python3-config --extension-suffix)

.PHONY: build
build: $(MODPATH)

.PHONY: run
run: build
	env PYTHONPATH=$(SETUPROOT) python3 -c \
		"import modbuf; print(modbuf.SimpleArrayFloat64(3).ndarray)"

$(SETUPROOT)/wrap_ConcreteBuffer.o: $(SETUPROOT)/modmesh/buffer/pymod/wrap_ConcreteBuffer.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArray.o: $(SETUPROOT)/modmesh/buffer/pymod/wrap_SimpleArray.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArrayPlex.o: $(SETUPROOT)/modmesh/buffer/pymod/wrap_SimpleArrayPlex.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/buffer_pymod.o: $(SETUPROOT)/modmesh/buffer/pymod/buffer_pymod.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/SimpleArray.o: $(SETUPROOT)/modmesh/buffer/SimpleArray.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/modbuf.o: modbuf.cpp
	c++ $(CFLAGS) -c $< -o $@

$(MODPATH): Makefile $(SETUPROOT)/SimpleArray.o $(SETUPROOT)/wrap_ConcreteBuffer.o $(SETUPROOT)/wrap_SimpleArray.o $(SETUPROOT)/wrap_SimpleArrayPlex.o $(SETUPROOT)/buffer_pymod.o $(SETUPROOT)/modbuf.o
	c++ $(LDFLAGS) \
		$(SETUPROOT)/SimpleArray.o \
		$(SETUPROOT)/wrap_ConcreteBuffer.o \
		$(SETUPROOT)/wrap_SimpleArray.o \
		$(SETUPROOT)/wrap_SimpleArrayPlex.o \
		$(SETUPROOT)/buffer_pymod.o \
		$(SETUPROOT)/modbuf.o \
		-o $@

.PHONY: copy
copy:
	@echo "BUILD_PATH": $(BUILD_PATH)
	@echo "standalone buffer root": $(SETUPROOT)
	rm -rf $(SETUPROOT)/modmesh
	mkdir -p $(SETUPROOT)/modmesh
	cp $(MMROOT)/cpp/modmesh/base.hpp $(SETUPROOT)/modmesh/base.hpp
	cp -a $(MMROOT)/cpp/modmesh/buffer $(SETUPROOT)/modmesh/buffer
	mkdir -p $(SETUPROOT)/modmesh/python
	cp $(MMROOT)/cpp/modmesh/python/common.hpp $(SETUPROOT)/modmesh/python/common.hpp
	cp -a $(MMROOT)/cpp/modmesh/toggle $(SETUPROOT)/modmesh/toggle
	rm -rf $(SETUPROOT)/modmesh/toggle/pymod
	cp $(MMROOT)/cpp/modmesh/base.hpp $(SETUPROOT)/modmesh/
	find $(SETUPROOT) -name CMakeLists.txt -delete