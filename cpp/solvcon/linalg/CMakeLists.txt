# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_LINALG_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/linalg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factorization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lu_factorization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kalman_filter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EigenSystem.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_LINALG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/EigenSystem.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_LINALG_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/linalg_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_LINALG_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/linalg_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_factorization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_kalman_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_EigenSystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_LuFactorization.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_LINALG_FILES
    ${SOLVCON_LINALG_HEADERS}
    ${SOLVCON_LINALG_SOURCES}
    ${SOLVCON_LINALG_PYMODHEADERS}
    ${SOLVCON_LINALG_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
