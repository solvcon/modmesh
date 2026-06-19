# Copyright (c) 2026, modmesh contributors
# BSD-style license; see COPYING

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(GPROF_EXECUTABLE gprof)
endif()

if(GPROF_EXECUTABLE)
    add_subdirectory(cprof)
endif()

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
