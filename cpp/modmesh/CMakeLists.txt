# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

if (BUILD_QT)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    find_package(Qt6 QUIET)
    if(NOT Qt6_FOUND)
        if(MSVC)
            message(FATAL_ERROR  "Qt6 Not Found.\nTry to add `C:\\Qt\\6.x.y\\msvcYYYY_64\\lib\\cmake` into \"Path\" of \"Enviroment Variables\"?")
        endif()
        if(APPLE)
            message(FATAL_ERROR  "Qt6 Not Found.\nTry `CMAKE_PREFIX_PATH=/path/to/qt/6.x.y/macos`?")
        endif()
        if(UNIX AND NOT APPLE)
            message(FATAL_ERROR  "Qt6 Not Found.\nTry `cmake -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x.y/gcc_64/lib/cmake`?")
        endif()
        message(FATAL_ERROR  "Qt6 Not Found.") # any other platforms
    endif()
    find_package(Qt6 REQUIRED COMPONENTS Core)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
    find_package(Qt6 REQUIRED COMPONENTS Gui)
    find_package(Qt6 REQUIRED COMPONENTS 3DCore)
    find_package(Qt6 REQUIRED COMPONENTS 3DRender)
    find_package(Qt6 REQUIRED COMPONENTS 3DInput)
    find_package(Qt6 REQUIRED COMPONENTS 3DExtras)
endif () # BUILD_QT

add_subdirectory(buffer)
add_subdirectory(mesh)
add_subdirectory(toggle)
add_subdirectory(bernstein)
add_subdirectory(onedim)
add_subdirectory(python)
add_subdirectory(spacetime)
add_subdirectory(view)
add_subdirectory(inout)

set(MODMESH_ROOT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/modmesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_ROOT_FILES
    ${MODMESH_ROOT_HEADERS}
    CACHE FILEPATH "" FORCE)

if (BUILD_METAL)
    add_subdirectory(device/metal)
    set(MODMESH_DEVICE_HEADERS
        ${MODMESH_DEVICE_HEADERS}
        ${MODMESH_METAL_HEADERS}
        CACHE FILEPATH "" FORCE)
    set(MODMESH_DEVICE_SOURCES
        ${MODMESH_DEVICE_SOURCES}
        ${MODMESH_METAL_SOURCES}
        CACHE FILEPATH "" FORCE)
else () # BUILD_METAL
    set(MODMESH_DEVICE_HEADERS
        CACHE FILEPATH "" FORCE)
    set(MODMESH_DEVICE_SOURCES
        CACHE FILEPATH "" FORCE)
endif () #BUILD_METAL

set(MODMESH_DEVICE_FILES
    ${MODMESH_DEVICE_HEADERS}
    ${MODMESH_DEVICE_SOURCES}
    CACHE FILEPATH "" FORCE)

set(MODMESH_TERMINAL_FILES
    ${MODMESH_ROOT_SOURCES}
    ${MODMESH_BUFFER_FILES}
    ${MODMESH_TOGGLE_FILES}
    ${MODMESH_BERNSTEIN_FILES}
    ${MODMESH_MESH_FILES}
    ${MODMESH_DEVICE_FILES}
    ${MODMESH_ONEDIM_FILES}
    ${MODMESH_SPACETIME_FILES}
    ${MODMESH_PYTHON_FILES}
    ${MODMESH_INOUT_FILES}
    CACHE FILEPATH "" FORCE)

set(MODMESH_GRAPHIC_FILES
    ${MODMESH_VIEW_FILES}
    CACHE FILEPATH "" FORCE)

if (BUILD_QT)
    qt_add_library(
        modmesh_primary
        STATIC
        ${MODMESH_TERMINAL_FILES}
        ${MODMESH_GRAPHIC_FILES}
    )

    target_link_libraries(
        modmesh_primary PUBLIC
        Qt::Widgets
        Qt::3DCore
        Qt::3DExtras
        Qt::3DInput
        Qt::3DRender
        Qt::Core
        Qt::Gui
    )

    target_link_libraries(
        modmesh_primary PUBLIC
        ${PYSIDE6_LIBFILE}
        ${SHIBOKEN6_LIBFILE}
    )
else () # BUILD_QT
    add_library(
        modmesh_primary
        STATIC
        ${MODMESH_TERMINAL_FILES}
    )
endif () # BUILD_QT

set_target_properties(modmesh_primary PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (CLANG_TIDY_EXE AND USE_CLANG_TIDY)
    set_target_properties(
        modmesh_primary PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif () # CLANG_TIDY

# Hack: add a .clang-tidy file in the generated .rcc directory.
# See https://gitlab.kitware.com/cmake/cmake/-/merge_requests/777
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/.rcc/.clang-tidy" "---
Checks: '-*,llvm-twine-local'")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/modmesh_primary_autogen/.clang-tidy" "---
Checks: '-bugprone-suspicious-include,llvm-twine-local'")

if (APPLE)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
    find_library(APPLE_FWK_METAL Metal REQUIRED)

    target_link_libraries(
        modmesh_primary PUBLIC
        ${APPLE_FWK_FOUNDATION}
        ${APPLE_FWK_QUARTZ_CORE}
        ${APPLE_FWK_METAL}
    )
endif () # APPLE

if (MSVC)
    target_compile_options(
        modmesh_primary PRIVATE
        ${COMMON_COMPILER_OPTIONS}
        /bigobj # C1128: number of sections exceeded object file format limit
    )
else () # MSVC
    target_compile_options(
        modmesh_primary PRIVATE
        ${COMMON_COMPILER_OPTIONS}
        -Wno-unused-value # for PYBIND11_EXPAND_SIDE_EFFECTS in pybind11.h
        -Wno-noexcept-type # GCC
    )
endif () # MSVC

if(QT3D_USE_RHI)
    target_compile_options(modmesh_primary PRIVATE
        ${COMMON_COMPILER_OPTIONS}
        -DQT3D_USE_RHI
    )
endif() # QT3D_USE_RHI

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
