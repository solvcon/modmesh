# Copyright (c) 2025, Kuan-Hsien Lee <khlee870529@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.16)

set(MODMESH_SIMD_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/simd.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simd_support.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SIMD_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/simd_support.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SIMD_NEONHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/neon/neon_alias.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/neon/neon_type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/neon/neon.hpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SIMD_NEONSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/neon/neon_alias.cpp
    CACHE FILEPATH "" FORCE)

set(MODMESH_SIMD_FILES
    ${MODMESH_SIMD_HEADERS}
    ${MODMESH_SIMD_SOURCES}
    ${MODMESH_SIMD_NEONHEADERS}
    ${MODMESH_SIMD_NEONSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
