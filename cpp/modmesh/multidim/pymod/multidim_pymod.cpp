/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/multidim/pymod/multidim_pymod.hpp> // Must be the first include for Python.
#include <pybind11/stl.h>

#include <modmesh/modmesh.hpp>
#include <modmesh/python/common.hpp>

namespace modmesh
{

namespace python
{

struct multidim_pymod_tag
{
};

template <>
OneTimeInitializer<multidim_pymod_tag> & OneTimeInitializer<multidim_pymod_tag>::me()
{
    static OneTimeInitializer<multidim_pymod_tag> instance;
    return instance;
}

void initialize_multidim(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_multidim(mod);
    };

    OneTimeInitializer<multidim_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
