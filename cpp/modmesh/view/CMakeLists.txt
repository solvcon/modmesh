# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_VIEW_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/PythonInterpreter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/R3DWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMainWindow.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonText.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RStaticMesh.hpp
    CACHE FILEPATH ""
)

set(MODMESH_VIEW_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/PythonInterpreter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/R3DWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMainWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonText.cpp
    CACHE FILEPATH ""
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
