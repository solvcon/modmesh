# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)
project(viewer LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# get the root of the modmdesh project
set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR}/../../../)

if(NOT DEFINED INSTALL_VIEWERDIR)
    set(INSTALL_VIEWERDIR "viewer")
endif()

find_package(Qt6 COMPONENTS Core)
find_package(Qt6 COMPONENTS Widgets)
find_package(Qt6 COMPONENTS Gui)
find_package(Qt6 COMPONENTS 3DCore)
find_package(Qt6 COMPONENTS 3DRender)
find_package(Qt6 COMPONENTS 3DInput)
find_package(Qt6 COMPONENTS 3DExtras)

qt_add_executable(
    viewer
    viewer.cpp
    ${MODMESH_HEADERS}
    ${MODMESH_SOURCES}
    ${MODMESH_VIEW_HEADERS}
    ${MODMESH_VIEW_SOURCES}
    ${MODMESH_ONEDIM_HEADERS}
    ${MODMESH_ONEDIM_SOURCES}
    ${MODMESH_SPACETIME_HEADERS}
    ${MODMESH_SPACETIME_SOURCES}
    ${MODMESH_PYTHON_HEADERS}
    ${MODMESH_PYTHON_SOURCES}
    ${MODMESH_PYMOD_MODMESH_HEADERS}
    ${MODMESH_PYMOD_MODMESH_SOURCES}
    ${MODMESH_PYMOD_ONEDIM_HEADERS}
    ${MODMESH_PYMOD_ONEDIM_SOURCES}
    ${MODMESH_PYMOD_SPACETIME_HEADERS}
    ${MODMESH_PYMOD_SPACETIME_SOURCES}
)

add_custom_target(run_viewer_pytest $<TARGET_FILE:viewer> --mode=pytest)

qt_add_resources(
    viewer "app_icon"
    PREFIX "/"
    BASE ${PROJECT_ROOT_DIR}/resources/viewer
    FILES
    ${PROJECT_ROOT_DIR}/resources/viewer/icon.ico
)

if(APPLE)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
    find_library(APPLE_FWK_METAL Metal REQUIRED)

    target_link_libraries(
        viewer PUBLIC
        ${APPLE_FWK_FOUNDATION}
        ${APPLE_FWK_QUARTZ_CORE}
        ${APPLE_FWK_METAL}
    )
endif()

if(HIDE_SYMBOL)
    set_target_properties(viewer PROPERTIES CXX_VISIBILITY_PRESET "hidden")
else()
    set_target_properties(viewer PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

target_compile_options(
    viewer PRIVATE
    ${COMMON_COMPILER_OPTIONS}
)

if(CLANG_TIDY_EXE AND USE_CLANG_TIDY)
    set_target_properties(
        viewer PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif()

# Hack: add a .clang-tidy file in the generated .rcc directory.
# See https://gitlab.kitware.com/cmake/cmake/-/merge_requests/777
file(WRITE "${viewer_BINARY_DIR}/.rcc/.clang-tidy" "---
Checks: '-*,llvm-twine-local'")
file(WRITE "${viewer_BINARY_DIR}/viewer_autogen/.clang-tidy" "---
Checks: '-bugprone-suspicious-include,llvm-twine-local'")

set_target_properties(
    viewer PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
)

target_link_libraries(
    viewer PUBLIC
    pybind11::embed
    Qt::Widgets
    Qt::3DCore
    Qt::3DExtras
    Qt::3DInput
    Qt::3DRender
    Qt::Core
    Qt::Gui
)

install(TARGETS viewer
    RUNTIME DESTINATION "${INSTALL_VIEWERDIR}"
    BUNDLE DESTINATION "${INSTALL_VIEWERDIR}"
    LIBRARY DESTINATION "${INSTALL_VIEWERDIR}"
)