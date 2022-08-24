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
message(STATUS "BUILD_QT: ${BUILD_QT}")

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
        set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-header-filter=/cpp/.\*/modmesh/.\*")
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
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands")

option(HIDE_SYMBOL "hide the symbols of python wrapper" OFF)
message(STATUS "HIDE_SYMBOL: ${HIDE_SYMBOL}")

option(DEBUG_SYMBOL "add debug information" ON)
message(STATUS "DEBUG_SYMBOL: ${DEBUG_SYMBOL}")
if(DEBUG_SYMBOL)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEBUG")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    endif()
endif()

option(MODMESH_PROFILE "enable profiler" OFF)
message(STATUS "MODMESH_PROFILE: ${MODMESH_PROFILE}")
if(MODMESH_PROFILE)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DMODMESH_PROFILE")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMODMESH_PROFILE")
    endif()
endif()

if(BUILD_METAL)
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

set(MODMESH_USE_PYSIDE False CACHE BOOL "Choose to build with pyside or not")
message(STATUS "MODMESH_USE_PYSIDE: ${MODMESH_USE_PYSIDE}")

if (MODMESH_USE_PYSIDE)
    add_compile_definitions(MODMESH_USE_PYSIDE)
endif () # MODMESH_USE_PYSIDE

execute_process(
    COMMAND python3 -c "import sys, os, PySide6; sys.stdout.write(os.path.dirname(PySide6.__file__))"
    OUTPUT_VARIABLE PYSIDE6_PYTHON_PACKAGE_PATH
)
message(STATUS "PYSIDE6_PYTHON_PACKAGE_PATH: ${PYSIDE6_PYTHON_PACKAGE_PATH}")

execute_process(
    COMMAND python3 -c "import sys, os, shiboken6; sys.stdout.write(os.path.dirname(shiboken6.__file__))"
    OUTPUT_VARIABLE SHIBOKEN6_PYTHON_PACKAGE_PATH
)
message(STATUS "SHIBOKEN6_PYTHON_PACKAGE_PATH: ${SHIBOKEN6_PYTHON_PACKAGE_PATH}")
execute_process(
    COMMAND python3 -c "import sys, os, shiboken6_generator; sys.stdout.write(os.path.dirname(shiboken6_generator.__file__))"
    OUTPUT_VARIABLE SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH
)
message(STATUS "SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH: ${SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH}")

if(NOT "${PYSIDE6_PYTHON_PACKAGE_PATH}" STREQUAL "")
    include_directories("${PYSIDE6_PYTHON_PACKAGE_PATH}/include")
    link_directories("${PYSIDE6_PYTHON_PACKAGE_PATH}")
endif()
if(NOT "${SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH}" STREQUAL "")
    include_directories("${SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH}/include")
endif()
if(NOT "${SHIBOKEN6_PYTHON_PACKAGE_PATH}" STREQUAL "")
    link_directories("${SHIBOKEN6_PYTHON_PACKAGE_PATH}")
endif()

file(GLOB PYSIDE6_LIBFILE LIST_DIRECTORIES false "${PYSIDE6_PYTHON_PACKAGE_PATH}/libpyside6.*")
file(GLOB SHIBOKEN6_LIBFILE LIST_DIRECTORIES false "${SHIBOKEN6_PYTHON_PACKAGE_PATH}/libshiboken6.*")

add_subdirectory(cpp/modmesh)

include(GNUInstallDirs)

set(MODMESH_PY_DIR "${PROJECT_SOURCE_DIR}/modmesh")

if(MSVC)
    set(COMMON_COMPILER_OPTIONS /W4 /bigobj /WX)
else()
    set(COMMON_COMPILER_OPTIONS -Wall -Wextra)
    if(USE_CLANG_TIDY AND LINT_AS_ERRORS)
        message(STATUS "Disable -Werror because clang-tidy (with lint as errors) is enabled")
    else()
        set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -Werror)
        message(STATUS "Enable -Werror because clang-tidy (with lint as errors) is disabled")
    endif()
endif()

add_subdirectory(cpp/binary/pymod_modmesh)
if(BUILD_QT)
add_subdirectory(cpp/binary/viewer)
set(ENV{QT3D_RENDERER} ${QT3D_RENDERER})
message(STATUS "QT3D_RENDERER: " $ENV{QT3D_RENDERER})
endif()

add_custom_target(flake8)
flake8("flake8")

set(USE_GOOGLETEST True CACHE BOOL "Build with googletest")
message(STATUS "USE_GOOGLETEST: ${USE_GOOGLETEST}")

if(USE_GOOGLETEST)
add_subdirectory(gtests)
endif() # USE_GOOGLETEST

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
