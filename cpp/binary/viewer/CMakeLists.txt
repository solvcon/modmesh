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
    ${MODMESH_PYTHON_HEADERS}
    ${MODMESH_PYTHON_SOURCES}
    ${MODMESH_PYMOD_MODMESH_HEADERS}
    ${MODMESH_PYMOD_MODMESH_SOURCES}
    ${MODMESH_PYMOD_VIEW_HEADERS}
    ${MODMESH_PYMOD_VIEW_SOURCES}
)

qt_add_resources(viewer "app_icon"
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
