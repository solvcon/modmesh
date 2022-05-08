# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

add_subdirectory(python/wrapper/modmesh)
add_subdirectory(python/wrapper/view)
add_subdirectory(view)

set(MODMESH_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/modmesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profile.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh/mesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh/StaticMesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/small_vector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/ConcreteBuffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer/SimpleArray.hpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh/StaticMesh_interior.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh/StaticMesh_boundary.cpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_PY_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/python.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/common.hpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_PY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/python/python.cpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
