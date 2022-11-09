# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)
project(viewer LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# get the root of the modmdesh project
set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR}/../../../)

if(NOT DEFINED INSTALL_VIEWERDIR)
    set(INSTALL_VIEWERDIR "viewer")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

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
)

target_link_libraries(
    viewer PUBLIC
    pybind11::embed
    modmesh_primary
)
if(WIN32)
    set_target_properties(
        viewer
        PROPERTIES
        LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE" # open the console for debug mode
    )
endif(WIN32)

add_custom_target(run_viewer_pytest $<TARGET_FILE:viewer> --mode=pytest)

qt_add_resources(
    viewer "app_icon"
    PREFIX "/"
    BASE ${PROJECT_ROOT_DIR}/resources/viewer
    FILES
    ${PROJECT_ROOT_DIR}/resources/viewer/icon.ico
)

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

add_custom_target(run_viewer_pytest $<TARGET_FILE:viewer> --mode=pytest)

install(TARGETS viewer
    RUNTIME DESTINATION "${INSTALL_VIEWERDIR}"
    BUNDLE DESTINATION "${INSTALL_VIEWERDIR}"
    LIBRARY DESTINATION "${INSTALL_VIEWERDIR}"
)