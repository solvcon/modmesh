# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_PYMOD_MODMESH_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/modmesh.hpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_PYMOD_MODMESH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/modmesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TypeBroadcast.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_profile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_ConcreteBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_SimpleArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_StaticGrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_StaticMesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_Toggle.cpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
