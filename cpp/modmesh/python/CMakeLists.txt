# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

add_subdirectory(wrapper/modmesh)
add_subdirectory(wrapper/onedim)
add_subdirectory(wrapper/spacetime)

set(MODMESH_PYTHON_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/python.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_PYTHON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
