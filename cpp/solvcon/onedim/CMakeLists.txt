# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_ONEDIM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Euler1DCore.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/onedim.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_ONEDIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Euler1DCore.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_ONEDIM_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/onedim_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_ONEDIM_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/onedim_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_onedim.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_ONEDIM_FILES
    ${SOLVCON_ONEDIM_HEADERS}
    ${SOLVCON_ONEDIM_SOURCES}
    ${SOLVCON_ONEDIM_PYMODHEADERS}
    ${SOLVCON_ONEDIM_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
