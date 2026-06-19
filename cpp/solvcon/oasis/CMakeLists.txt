# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_OASIS_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/oasis_device.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_OASIS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/oasis_device.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_OASIS_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/oasis_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_OASIS_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/oasis_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_oasis.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_OASIS_FILES
    ${SOLVCON_OASIS_HEADERS}
    ${SOLVCON_OASIS_SOURCES}
    ${SOLVCON_OASIS_PYMODHEADERS}
    ${SOLVCON_OASIS_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
