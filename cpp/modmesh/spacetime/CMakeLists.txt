# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.16)

set(MODMESH_SPACETIME_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Celm.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Celm_decl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ElementBase.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ElementBase_decl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Field.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Field_decl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Grid.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Grid_decl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Selm.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Selm_decl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Solver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase_decl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spacetime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_spacetime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/inviscid_burgers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/linear_scalar.hpp
    CACHE FILEPATH "" FORCE
)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
