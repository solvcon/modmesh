# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(MODMESH_SIMD_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/simd.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simd_support.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SIMD_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/simd_support.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SIMD_NEONHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/neon/neon_alias.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/neon/neon_type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/neon/neon.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SIMD_NEONSOURCES
    CACHE FILEPATH "" FORCE)

set(MODMESH_SIMD_FILES
    ${MODMESH_SIMD_HEADERS}
    ${MODMESH_SIMD_SOURCES}
    ${MODMESH_SIMD_NEONHEADERS}
    ${MODMESH_SIMD_NEONSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
