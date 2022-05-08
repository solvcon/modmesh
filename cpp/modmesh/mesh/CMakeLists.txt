# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_MESH_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StaticMesh.hpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_MESH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/StaticMesh_boundary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StaticMesh_interior.cpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
