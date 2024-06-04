# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

cmake_minimum_required(VERSION 3.24)

# See https://google.github.io/googletest/quickstart-cmake.html for the
# recommended cmake configuration
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    test_nopython
    test_nopython_buffer.cpp
    test_nopython_modmesh.cpp
    test_nopython_inout.cpp
    test_nopython_radixtree.cpp
    test_nopython_callprofiler.cpp
    test_nopython_callprofilerserializer.cpp
    ${MODMESH_TOGGLE_SOURCES}
    ${MODMESH_BUFFER_SOURCES}
)
target_link_libraries(
    test_nopython
    GTest::gtest_main
    GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(test_nopython)

add_custom_target(run_gtest
    COMMAND $<TARGET_FILE:test_nopython>
    DEPENDS test_nopython)
