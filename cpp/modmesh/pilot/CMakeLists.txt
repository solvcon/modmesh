# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_PILOT_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/R3DWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RWorld.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAxisMark.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RCameraController.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonConsoleDockWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RStaticMesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAction.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pilot.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_pilot.hpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_PILOT_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/R3DWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RWorld.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAxisMark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RCameraController.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonConsoleDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RStaticMesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_pilot.cpp
    CACHE FILEPATH "" FORCE
)

set(MODMESH_PILOT_FILES
    ${MODMESH_PILOT_PYMODHEADERS}
    ${MODMESH_PILOT_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
