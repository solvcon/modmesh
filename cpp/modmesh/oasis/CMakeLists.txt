# Copyright (c) 2026, Han-Xuan Huang <c1ydehhx@gmail.com>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(MODMESH_OASIS_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/oasis_device.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_OASIS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/oasis_device.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_OASIS_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/oasis_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_OASIS_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/oasis_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_oasis.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_OASIS_FILES
    ${MODMESH_OASIS_HEADERS}
    ${MODMESH_OASIS_SOURCES}
    ${MODMESH_OASIS_PYMODHEADERS}
    ${MODMESH_OASIS_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
