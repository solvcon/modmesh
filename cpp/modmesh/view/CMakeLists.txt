# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_VIEW_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/R3DWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RApplication.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAxisMark.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMainWindow.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonConsoleDockWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RStaticMesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAction.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/view.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_view.hpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_VIEW_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/R3DWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAxisMark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMainWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonConsoleDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RStaticMesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_view.cpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
