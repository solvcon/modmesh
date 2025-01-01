cmake_minimum_required(VERSION 3.16)

set(MODMESH_MATHTYPE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mathtype.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Complex.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_MATHTYPE_FILES
    ${MODMESH_MATHTYPE_HEADERS}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
