/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/buffer/pymod/buffer_pymod.hpp> // Must be the first include.

#include <modmesh/simd/simd_support.hpp>

namespace modmesh
{

namespace python
{

namespace
{

char const * simd_feature_name()
{
    using namespace modmesh::simd::detail;
    switch (detect_simd())
    {
    case SIMD_NONE: return "NONE";
    case SIMD_NEON: return "NEON";
    case SIMD_SSE: return "SSE";
    case SIMD_SSE2: return "SSE2";
    case SIMD_SSE3: return "SSE3";
    case SIMD_SSSE3: return "SSSE3";
    case SIMD_SSE41: return "SSE41";
    case SIMD_SSE42: return "SSE42";
    case SIMD_AVX: return "AVX";
    case SIMD_AVX2: return "AVX2";
    case SIMD_AVX512: return "AVX512";
    case SIMD_UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

} // namespace

struct buffer_pymod_tag;

template <>
OneTimeInitializer<buffer_pymod_tag> & OneTimeInitializer<buffer_pymod_tag>::me()
{
    static OneTimeInitializer<buffer_pymod_tag> instance;
    return instance;
}

void initialize_buffer(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        import_numpy();

        wrap_ConcreteBuffer(mod);
        wrap_SimpleArray(mod);
        wrap_SimpleArrayPlex(mod);

        // Reports the runtime-detected SIMD feature so pytest can verify that
        // NEON dispatch is active on aarch64. Without this guard, a regression
        // that silently routes everything to the scalar path would still pass
        // every correctness check. Kept under an underscore-prefixed name
        // because detect_simd() only meaningfully reflects the dispatched
        // backend on aarch64 today; on other targets it would mislead users.
        mod.def("_simd_feature", &simd_feature_name);
    };

    OneTimeInitializer<buffer_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
