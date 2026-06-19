/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/universe/pymod/universe_pymod.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

struct bernstein_pymod_tag;

template <>
OneTimeInitializer<bernstein_pymod_tag> & OneTimeInitializer<bernstein_pymod_tag>::me()
{
    static OneTimeInitializer<bernstein_pymod_tag> instance;
    return instance;
}

void initialize_universe(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_shape0d(mod);
        wrap_shape1d(mod);
        wrap_shape2d(mod);
        wrap_shape3d(mod);
        wrap_view_transform2d(mod);
        wrap_World(mod);
    };

    OneTimeInitializer<bernstein_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: