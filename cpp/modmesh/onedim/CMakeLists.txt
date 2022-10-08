# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_ONEDIM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Euler1DCore.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/onedim.hpp
    CACHE FILEPATH "" FORCE
    )

set(MODMESH_ONEDIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Euler1DCore.cpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
