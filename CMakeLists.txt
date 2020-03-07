# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.9)
project(modmesh)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

option(USE_CLANG_TIDY "use clang-tidy" OFF)
option(LINT_AS_ERRORS "clang-tidy warnings as errors" OFF)

find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "Path to clang-tidy executable"
)
if(USE_CLANG_TIDY)
    if(CLANG_TIDY_EXE)
        set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}")
        if(LINT_AS_ERRORS)
            set(DO_CLANG_TIDY "${DO_CLANG_TIDY}" "-warnings-as-errors=*")
        endif()
        message(STATUS "use clang-tidy with DO_CLANG_TIDY: ${DO_CLANG_TIDY}")
    else()
        message(FATAL_ERROR "USE_CLANG_TIDY is on but clang-tidy is not found")
    endif()
else()
    message(STATUS "not use clang-tidy")
endif()

include(Flake8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands")

option(HIDE_SYMBOL "hide the symbols of python wrapper" OFF)
message(STATUS "HIDE_SYMBOL: ${HIDE_SYMBOL}")

option(DEBUG_SYMBOL "add debug information" ON)
message(STATUS "DEBUG_SYMBOL: ${DEBUG_SYMBOL}")
if(DEBUG_SYMBOL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

option(MODMESH_PROFILE "enable profiler" OFF)
message(STATUS "MODMESH_PROFILE: ${MODMESH_PROFILE}")
if(MODMESH_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMODMESH_PROFILE")
endif()

option(pybind11_path "pybind11 path")
find_package(pybind11 REQUIRED PATHS ${pybind11_path})
message(STATUS "pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")
include_directories(${pybind11_INCLUDE_DIRS})

find_package(NumPy REQUIRED)
message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
include_directories(${NUMPY_INCLUDE_DIR})

set(MODMESH_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "")

include_directories(${MODMESH_INCLUDE_DIR})

set(MODMESH_HEADERS
    include/modmesh/modmesh.hpp
    include/modmesh/base.hpp
    include/modmesh/profile.hpp
    include/modmesh/grid.hpp
    include/modmesh/small_vector.hpp
    include/modmesh/ConcreteBuffer.hpp
    include/modmesh/SimpleArray.hpp
)
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/"
       MODMESH_HEADERS "${MODMESH_HEADERS}")

set(MODMESH_PY_HEADERS
    include/modmesh/python/python.hpp
    include/modmesh/python/common.hpp
)
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/"
       MODMESH_PY_HEADERS "${MODMESH_PY_HEADERS}")

include(GNUInstallDirs)

set(MODMESH_SOURCES
    src/modmesh.cpp
)
pybind11_add_module(
    _modmesh
    ${MODMESH_SOURCES}
    ${MODMESH_HEADERS}
    ${MODMESH_PY_HEADERS}
)
if(HIDE_SYMBOL)
    set_target_properties(_modmesh PROPERTIES CXX_VISIBILITY_PRESET "hidden")
else()
    set_target_properties(_modmesh PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

if(CLANG_TIDY_EXE AND USE_CLANG_TIDY)
    set_target_properties(
        _modmesh PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif()

set(MODMESH_PY_PATH "${PROJECT_SOURCE_DIR}/modmesh")

add_custom_target(_modmesh_py
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_modmesh> ${MODMESH_PY_PATH}
    DEPENDS _modmesh)

message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY ${MODMESH_INCLUDE_DIR}/modmesh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_custom_target(flake8)
flake8("flake8")

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
