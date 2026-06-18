/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/toggle/pymod/toggle_pymod.hpp> // Must be the first include.
#include <modmesh/buffer/pymod/buffer_pymod.hpp>

namespace modmesh
{

namespace python
{

struct toggle_pymod_tag;

template <>
OneTimeInitializer<toggle_pymod_tag> & OneTimeInitializer<toggle_pymod_tag>::me()
{
    static OneTimeInitializer<toggle_pymod_tag> instance;
    return instance;
}

void initialize_toggle(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_profile(mod);
        wrap_Toggle(mod);
    };

    OneTimeInitializer<toggle_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
