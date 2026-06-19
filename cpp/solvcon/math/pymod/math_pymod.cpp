/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/pymod/math_pymod.hpp>

namespace solvcon
{

namespace python
{

struct math_pymod_tag;

template <>
OneTimeInitializer<math_pymod_tag> & OneTimeInitializer<math_pymod_tag>::me()
{
    static OneTimeInitializer<math_pymod_tag> instance;
    return instance;
}

void initialize_math(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_Complex(mod);
    };

    OneTimeInitializer<math_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
