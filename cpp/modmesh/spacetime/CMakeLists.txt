# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_SPACETIME_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spacetime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/BadEuler1DSolver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/inviscid_burgers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/linear_scalar.hpp
    CACHE FILEPATH "" FORCE
    )

set(MODMESH_SPACETIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/BadEuler1DSolver.cpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
