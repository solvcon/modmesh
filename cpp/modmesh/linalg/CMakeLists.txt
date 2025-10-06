# Copyright (c) 2025, Chun-Shih Chang <austin20463@gmail.com>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_LINALG_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/factorization.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_LINALG_SOURCES
    CACHE FILEPATH "" FORCE)

set(MODMESH_LINALG_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/linalg_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_LINALG_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/linalg_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_linalg.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_LINALG_FILES
    ${MODMESH_LINALG_HEADERS}
    ${MODMESH_LINALG_SOURCES}
    ${MODMESH_LINALG_PYMODHEADERS}
    ${MODMESH_LINALG_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
