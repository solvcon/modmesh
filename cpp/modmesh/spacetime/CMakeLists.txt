# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(MODMESH_SPACETIME_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spacetime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/inviscid_burgers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/linear_scalar.hpp
    CACHE FILEPATH "" FORCE
    )

set(MODMESH_SPACETIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SPACETIME_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/spacetime_pymod.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_spacetime.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SPACETIME_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/spacetime_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_spacetime.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SPACETIME_FILES
    ${MODMESH_SPACETIME_HEADERS}
    ${MODMESH_SPACETIME_SOURCES}
    ${MODMESH_SPACETIME_PYMODHEADERS}
    ${MODMESH_SPACETIME_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
