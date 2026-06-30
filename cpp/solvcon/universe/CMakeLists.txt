# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_UNIVERSE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/bernstein.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bezier.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/coord.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/polygon.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/polygon_boolean.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ViewTransform2d.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/World.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/WorldDiagnostics.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rtree.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_UNIVERSE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bernstein.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/World.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_UNIVERSE_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/universe_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_UNIVERSE_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/universe_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_shape0d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_shape1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_shape2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_shape3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_view_transform2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_World.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_UNIVERSE_FILES
    ${SOLVCON_UNIVERSE_HEADERS}
    ${SOLVCON_UNIVERSE_SOURCES}
    ${SOLVCON_UNIVERSE_PYMODHEADERS}
    ${SOLVCON_UNIVERSE_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
