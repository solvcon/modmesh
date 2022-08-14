# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_PYMOD_ONEDIM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/onedim.hpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_PYMOD_ONEDIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/onedim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_onedim.cpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
