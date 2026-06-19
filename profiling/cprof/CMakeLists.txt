# Copyright (c) 2026, modmesh contributors
# BSD-style license; see COPYING

set(CPROF_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(CPROF_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_workload.py")
set(CPROF_SHARD_COUNT 32)
set(CPROF_GENERATED_SOURCES
    "${CPROF_GENERATED_DIR}/callprofiler_workload.cpp"
    "${CPROF_GENERATED_DIR}/callprofiler_workload_functions.hpp")

math(EXPR CPROF_LAST_SHARD "${CPROF_SHARD_COUNT} - 1")
foreach(index RANGE 0 ${CPROF_LAST_SHARD})
    list(APPEND CPROF_GENERATED_SOURCES
        "${CPROF_GENERATED_DIR}/callprofiler_workload_${index}.cpp")
endforeach()

add_custom_command(
    OUTPUT ${CPROF_GENERATED_SOURCES}
    COMMAND "${PYTHON_EXECUTABLE}" "${CPROF_GENERATOR}"
        --output-dir "${CPROF_GENERATED_DIR}"
        --shards "${CPROF_SHARD_COUNT}"
    DEPENDS "${CPROF_GENERATOR}"
    VERBATIM
)

add_executable(
    callprofiler_gprof
    callprofiler_gprof.cpp
    callprofiler_workload.hpp
    ${CPROF_GENERATED_SOURCES}
    ${PROJECT_SOURCE_DIR}/cpp/modmesh/toggle/RadixTree.cpp
)

target_include_directories(
    callprofiler_gprof PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CPROF_GENERATED_DIR}"
)

target_compile_options(
    callprofiler_gprof PRIVATE
    ${COMMON_COMPILER_OPTIONS}
    -pg
    -fno-omit-frame-pointer
)

target_link_options(callprofiler_gprof PRIVATE -pg)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(callprofiler_gprof PRIVATE -no-pie)
endif()

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
