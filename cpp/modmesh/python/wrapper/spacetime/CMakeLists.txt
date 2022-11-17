# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_SPACETIME_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/spacetime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_spacetime.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SPACETIME_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/spacetime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_spacetime.cpp
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
