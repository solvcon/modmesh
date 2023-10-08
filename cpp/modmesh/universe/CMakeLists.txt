# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_UNIVERSE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/bernstein.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/World.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_UNIVERSE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bernstein.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/World.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_UNIVERSE_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/universe_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_UNIVERSE_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/universe_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_bernstein.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_World.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_UNIVERSE_FILES
    ${MODMESH_UNIVERSE_HEADERS}
    ${MODMESH_UNIVERSE_SOURCES}
    ${MODMESH_UNIVERSE_PYMODHEADERS}
    ${MODMESH_UNIVERSE_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
