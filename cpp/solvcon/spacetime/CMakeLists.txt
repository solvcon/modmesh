# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_SPACETIME_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spacetime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/inviscid_burgers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/linear_scalar.hpp
    CACHE FILEPATH "" FORCE
    )

set(SOLVCON_SPACETIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_SPACETIME_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/spacetime_pymod.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_spacetime.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_SPACETIME_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/spacetime_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_spacetime.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_SPACETIME_FILES
    ${SOLVCON_SPACETIME_HEADERS}
    ${SOLVCON_SPACETIME_SOURCES}
    ${SOLVCON_SPACETIME_PYMODHEADERS}
    ${SOLVCON_SPACETIME_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
