# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_MULTIDIM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/multidim.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GradientElement.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/euler.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MULTIDIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/GradientElement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/euler_ce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/euler_solution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/euler_march.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/euler_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/euler_bc.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MULTIDIM_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/multidim_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MULTIDIM_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/multidim_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_multidim.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MULTIDIM_FILES
    ${SOLVCON_MULTIDIM_HEADERS}
    ${SOLVCON_MULTIDIM_SOURCES}
    ${SOLVCON_MULTIDIM_PYMODHEADERS}
    ${SOLVCON_MULTIDIM_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
