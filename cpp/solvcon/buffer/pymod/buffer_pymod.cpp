/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/buffer_pymod.hpp> // Must be the first include.

#include <solvcon/simd/simd_support.hpp>

namespace solvcon
{

namespace python
{

static char const * simd_feature_name()
{
    namespace simd_detail = solvcon::simd::detail;
    switch (simd_detail::detect_simd())
    {
    case simd_detail::SIMD_NONE: return "NONE";
    case simd_detail::SIMD_NEON: return "NEON";
    case simd_detail::SIMD_SSE: return "SSE";
    case simd_detail::SIMD_SSE2: return "SSE2";
    case simd_detail::SIMD_SSE3: return "SSE3";
    case simd_detail::SIMD_SSSE3: return "SSSE3";
    case simd_detail::SIMD_SSE41: return "SSE41";
    case simd_detail::SIMD_SSE42: return "SSE42";
    case simd_detail::SIMD_AVX: return "AVX";
    case simd_detail::SIMD_AVX2: return "AVX2";
    case simd_detail::SIMD_AVX512: return "AVX512";
    case simd_detail::SIMD_UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

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

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
