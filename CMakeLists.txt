# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.27)
project(modmesh)

cmake_policy(SET CMP0148 OLD)

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

# Sanitizers
# - Undefined Behavior Sanitization: Preventing the unpredictable outcomes of undefined behavior. See https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
# - Memory Leaks Sanitization: Prevententing memory leaks caused from inproper memory management. See https://clang.llvm.org/docs/MemorySanitizer.html
# - Address Sanitization: Protecting against address-related vulnerabilities. See https://clang.llvm.org/docs/AddressSanitizer.html
option(USE_SANITIZER "use sanitizer (undefined, leak, address)" OFF)

option(USE_PYTEST_HELPER_BINDING "use helper bindings to run pytest. should be ON if the build is for pytest" OFF)
if(USE_PYTEST_HELPER_BINDING)
    add_definitions(-D USE_PYTEST_HELPER_BINDING)
endif()
message(STATUS "USE_PYTEST_HELPER_BINDING: ${USE_PYTEST_HELPER_BINDING}")


if(BUILD_QT)
    option(QT3D_USE_RHI "Qt use RHI" OFF)
    message(STATUS "QT3D_USE_RHI: ${QT3D_USE_RHI}")
endif()

# PUI-related code has been migrated to PySide6, so it doesn't need to be installed by default.
# Yet, you can still install it by setting the INSTALL_PUI option to ON.
option(INSTALL_PUI "install PUI" OFF)
message(STATUS "INSTALL_PUI : ${INSTALL_PUI}")
if(INSTALL_PUI)
    if(NOT BUILD_QT)
        message(WARNING "INSTALL_PUI ON while BUILD_QT OFF")
    endif()
    execute_process(
        COMMAND python3 PUI_install.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty
    )
endif()

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

if(USE_SANITIZER)
    message(STATUS "use sanitizer")
    set(SANITIZER_LIST "undefined,leak,address")
    add_compile_options(-fsanitize=${SANITIZER_LIST})
    add_compile_options(-DUSE_SANITIZER)
    add_link_options(-fsanitize=${SANITIZER_LIST})
else()
    message(STATUS "not use sanitizer")
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

# Pybind11 2.12.0 is the first version that supports compiling for numpy 2.0.0.
# Therefore, it is required to use pybind11 2.12.0 or later.
# Reference: https://github.com/pybind/pybind11/releases/tag/v2.12.0

option(pybind11_path "pybind11 path")
find_package(pybind11 2.12.0 REQUIRED PATHS ${pybind11_path})
message(STATUS "pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")
include_directories(${pybind11_INCLUDE_DIRS})

find_package(NumPy REQUIRED)
message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
include_directories(${NUMPY_INCLUDE_DIR})

set(MODMESH_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/cpp" CACHE INTERNAL "")

include_directories(${MODMESH_INCLUDE_DIR})

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

# PySide6 and Shiboken6 library name are different in Windows and Unix-like OS
# therefore needs sperate in different cases
if(MSVC)
    file(GLOB PYSIDE6_LIBFILE LIST_DIRECTORIES false "${PYSIDE6_PYTHON_PACKAGE_PATH}/pyside6.*.lib")
    file(GLOB SHIBOKEN6_LIBFILE LIST_DIRECTORIES false "${SHIBOKEN6_PYTHON_PACKAGE_PATH}/shiboken6.*.lib")
else()
    file(GLOB PYSIDE6_LIBFILE LIST_DIRECTORIES false "${PYSIDE6_PYTHON_PACKAGE_PATH}/libpyside6.*")
    file(GLOB SHIBOKEN6_LIBFILE LIST_DIRECTORIES false "${SHIBOKEN6_PYTHON_PACKAGE_PATH}/libshiboken6.*")
endif()

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
add_subdirectory(cpp/binary/pilot)
endif()

add_custom_target(flake8)
flake8("flake8")

set(USE_GOOGLETEST True CACHE BOOL "Build with googletest")
message(STATUS "USE_GOOGLETEST: ${USE_GOOGLETEST}")

if(USE_GOOGLETEST)
add_subdirectory(gtests)
endif() # USE_GOOGLETEST

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
