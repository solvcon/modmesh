# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 4.0.1)
project(modmesh_pymod)

pybind11_add_module(
    _modmesh
    module.cpp
)

target_link_libraries(
    _modmesh PUBLIC
    pybind11::module
    modmesh_primary
)

if (HIDE_SYMBOL)
    set_target_properties(_modmesh PROPERTIES CXX_VISIBILITY_PRESET "hidden")
else () # HIDE_SYMBOL
    set_target_properties(_modmesh PROPERTIES CXX_VISIBILITY_PRESET "default")
endif () # HIDE_SYMBOL

target_compile_options(
    _modmesh PRIVATE
    ${COMMON_COMPILER_OPTIONS}
)

if (CLANG_TIDY_EXE AND USE_CLANG_TIDY)
    set_target_properties(
        _modmesh PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif ()

include(CMakePrintHelpers)
get_filename_component(REALPATH_PYTHON "${PYTHON_EXECUTABLE}" REALPATH)
get_filename_component(PYTHON_VENV_BIN "${REALPATH_PYTHON}" DIRECTORY)
cmake_print_variables(REALPATH_PYTHON)
cmake_print_variables(PYTHON_VENV_BIN)

execute_process(
    COMMAND ${PYTHON_VENV_BIN}/python3-config --extension-suffix
    OUTPUT_VARIABLE PYEXTSUFFIX
)

add_custom_target(_modmesh_py
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_modmesh> ${MODMESH_PY_DIR}/../_modmesh${PYEXTSUFFIX}
    DEPENDS _modmesh)

message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY ${MODMESH_INCLUDE_DIR}/modmesh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
