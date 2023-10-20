# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_TOGGLE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/profile.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RadixTree.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/toggle.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_TOGGLE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/toggle.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_TOGGLE_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/toggle_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_TOGGLE_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/toggle_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_profile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_Toggle.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_TOGGLE_FILES
    ${MODMESH_TOGGLE_HEADERS}
    ${MODMESH_TOGGLE_SOURCES}
    ${MODMESH_TOGGLE_PYMODHEADERS}
    ${MODMESH_TOGGLE_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
