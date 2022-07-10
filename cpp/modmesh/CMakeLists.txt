# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

add_subdirectory(buffer)
add_subdirectory(mesh)
add_subdirectory(python)
add_subdirectory(spacetime)
add_subdirectory(view)

set(MODMESH_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/modmesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profile.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/toggle/toggle.hpp
    ${MODMESH_BUFFER_HEADERS}
    ${MODMESH_MESH_HEADERS}
    CACHE FILEPATH "" FORCE
)

set(MODMESH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/toggle/toggle.cpp
    ${MODMESH_MESH_SOURCES}
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
