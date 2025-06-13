cmake_minimum_required(VERSION 3.16)

set(MODMESH_TRANSFORM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fourier.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_TRANSFORM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/fourier.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_TRANSFORM_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/transform_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_fourier.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_TRANSFORM_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/transform_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_TRANSFORM_FILES
    ${MODMESH_TRANSFORM_HEADERS}
    ${MODMESH_TRANSFORM_SOURCES}
    ${MODMESH_TRANSFORM_PYMODSOURCES}
    ${MODMESH_TRANSFORM_PYMODHEADERS}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
