# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_BUFFER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/small_vector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConcreteBuffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleArray.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_BUFFER_SOURCES
    CACHE FILEPATH "" FORCE)

set(MODMESH_BUFFER_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/buffer_pymod.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/TypeBroadcast.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_BUFFER_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_ConcreteBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArray.cpp
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
