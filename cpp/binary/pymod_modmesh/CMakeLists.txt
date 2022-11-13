# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)
project(modmesh_pymod)

pybind11_add_module(
    _modmesh
    module.cpp
    ${MODMESH_PYMOD_MODMESH_HEADERS}
    ${MODMESH_PYMOD_MODMESH_SOURCES}
    ${MODMESH_PYMOD_ONEDIM_HEADERS}
    ${MODMESH_PYMOD_ONEDIM_SOURCES}
    ${MODMESH_PYMOD_SPACETIME_HEADERS}
    ${MODMESH_PYMOD_SPACETIME_SOURCES}
    ${MODMESH_HEADERS}
    ${MODMESH_SOURCES}
    ${MODMESH_ONEDIM_HEADERS}
    ${MODMESH_ONEDIM_SOURCES}
    ${MODMESH_SPACETIME_HEADERS}
    ${MODMESH_SPACETIME_SOURCES}
    ${MODMESH_PYTHON_HEADERS}
    ${MODMESH_PYTHON_SOURCES}
)

if (BUILD_QT)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    find_package(Qt6 REQUIRED COMPONENTS Core)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
    find_package(Qt6 REQUIRED COMPONENTS Gui)
    find_package(Qt6 REQUIRED COMPONENTS 3DCore)
    find_package(Qt6 REQUIRED COMPONENTS 3DRender)
    find_package(Qt6 REQUIRED COMPONENTS 3DInput)
    find_package(Qt6 REQUIRED COMPONENTS 3DExtras)

    qt_add_library(
        _modmesh_qt
        STATIC
        ${MODMESH_VIEW_HEADERS}
        ${MODMESH_VIEW_SOURCES}
    )

    target_link_libraries(
        _modmesh_qt PUBLIC
        pybind11::module
        Qt::Widgets
        Qt::3DCore
        Qt::3DExtras
        Qt::3DInput
        Qt::3DRender
        Qt::Core
        Qt::Gui
    )

    target_link_libraries(
        _modmesh PUBLIC
        _modmesh_qt
    )
endif () # BUILD_QT

if (APPLE)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
    find_library(APPLE_FWK_METAL Metal REQUIRED)

    target_link_libraries(
        _modmesh PUBLIC
        ${APPLE_FWK_FOUNDATION}
        ${APPLE_FWK_QUARTZ_CORE}
        ${APPLE_FWK_METAL}
    )
endif () # APPLE

if (HIDE_SYMBOL)
    set_target_properties(_modmesh PROPERTIES CXX_VISIBILITY_PRESET "hidden")
else () # HIDE_SYMBOL
    set_target_properties(_modmesh PROPERTIES CXX_VISIBILITY_PRESET "default")
endif () # HIDE_SYMBOL

if (MSVC)
    target_compile_options(
        _modmesh PRIVATE
        ${COMMON_COMPILER_OPTIONS}
    )
else () # MSVC
    target_compile_options(
        _modmesh PRIVATE
        ${COMMON_COMPILER_OPTIONS}
        -Wno-unused-value # for PYBIND11_EXPAND_SIDE_EFFECTS in pybind11.h
        -Wno-noexcept-type # GCC
    )
endif () # MSVC

if (CLANG_TIDY_EXE AND USE_CLANG_TIDY)
    set_target_properties(
        _modmesh PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif () # CLANG_TIDY

execute_process(
    COMMAND python3-config --extension-suffix
    OUTPUT_VARIABLE PYEXTSUFFIX
)

add_custom_target(_modmesh_py
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_modmesh> ${MODMESH_PY_DIR}/_modmesh${PYEXTSUFFIX}
    DEPENDS _modmesh)

message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY ${MODMESH_INCLUDE_DIR}/modmesh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
