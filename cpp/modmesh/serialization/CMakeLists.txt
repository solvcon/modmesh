# Copyright (c) 2024, An-Chi Liu <phy.tiger@gmail.com>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(MODMESH_SERIALIZATION_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/SerializableItem.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SERIALIZATION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/SerializableItem.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SERIALIZATION_FILES
    ${MODMESH_SERIALIZATION_HEADERS}
    ${MODMESH_SERIALIZATION_SOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
