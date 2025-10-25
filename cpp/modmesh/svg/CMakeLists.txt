# Copyright (c) 2025, Anchi Liu <phy.tiger@gmail.com>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_SVG_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/svg.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SVG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/svg.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SVG_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/svg_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SVG_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/svg_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_svg.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SVG_FILES
    ${MODMESH_SVG_HEADERS}
    ${MODMESH_SVG_SOURCES}
    ${MODMESH_SVG_PYMODHEADERS}
    ${MODMESH_SVG_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
