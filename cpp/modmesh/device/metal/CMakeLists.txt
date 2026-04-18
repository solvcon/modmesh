# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(MODMESH_METAL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/metal.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
