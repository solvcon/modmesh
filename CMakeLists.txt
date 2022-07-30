# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)
project(modmesh)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

if(NOT SKIP_PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE python3)
    message(STATUS "use PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
endif()

option(BUILD_QT "build with QT" ON)

option(BUILD_METAL "build with Metal" OFF)
message(STATUS "BUILD_METAL: ${BUILD_METAL}")
if(BUILD_METAL)
    add_compile_options(-DMODMESH_METAL)
endif()

option(USE_CLANG_TIDY "use clang-tidy" OFF)
option(LINT_AS_ERRORS "clang-tidy warnings as errors" OFF)

find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "Path to clang-tidy executable"
)
if(USE_CLANG_TIDY)
    if(CLANG_TIDY_EXE)
        set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}")
        if(LINT_AS_ERRORS)
            set(DO_CLANG_TIDY "${DO_CLANG_TIDY}" "-warnings-as-errors=*")
        endif()
        message(STATUS "use clang-tidy with DO_CLANG_TIDY: ${DO_CLANG_TIDY}")
    else()
        message(FATAL_ERROR "USE_CLANG_TIDY is on but clang-tidy is not found")
    endif()
else()
    message(STATUS "not use clang-tidy")
endif()

include(Flake8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands")

option(HIDE_SYMBOL "hide the symbols of python wrapper" OFF)
message(STATUS "HIDE_SYMBOL: ${HIDE_SYMBOL}")

option(DEBUG_SYMBOL "add debug information" ON)
message(STATUS "DEBUG_SYMBOL: ${DEBUG_SYMBOL}")
if(DEBUG_SYMBOL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

option(MODMESH_PROFILE "enable profiler" OFF)
message(STATUS "MODMESH_PROFILE: ${MODMESH_PROFILE}")
if(MODMESH_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMODMESH_PROFILE")
endif()

if(BUILD_QT)
include_directories("${CMAKE_CURRENT_LIST_DIR}/thirdparty/install/metal-cpp")
endif()

option(pybind11_path "pybind11 path")
find_package(pybind11 REQUIRED PATHS ${pybind11_path})
message(STATUS "pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")
include_directories(${pybind11_INCLUDE_DIRS})

find_package(NumPy REQUIRED)
message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
include_directories(${NUMPY_INCLUDE_DIR})

set(MODMESH_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/cpp" CACHE INTERNAL "")

include_directories(${MODMESH_INCLUDE_DIR})

add_subdirectory(cpp/modmesh)

include(GNUInstallDirs)

set(MODMESH_PY_DIR "${PROJECT_SOURCE_DIR}/modmesh")

set(COMMON_COMPILER_OPTIONS -Wall -Wextra)
if(USE_CLANG_TIDY AND LINT_AS_ERRORS)
    message(STATUS "Disable -Werror because clang-tidy (with lint as errors) is enabled")
else()
    set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -Werror)
    message(STATUS "Enable -Werror because clang-tidy (with lint as errors) is disabled")
endif()

add_subdirectory(cpp/binary/pymod_modmesh)
if(BUILD_QT)
add_subdirectory(cpp/binary/viewer)
endif()

add_custom_target(flake8)
flake8("flake8")

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
