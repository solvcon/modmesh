/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/transform/pymod/transform_pymod.hpp>

namespace modmesh
{

namespace python
{

struct transform_pymod_tag;

template <>
OneTimeInitializer<transform_pymod_tag> & OneTimeInitializer<transform_pymod_tag>::me()
{
    static OneTimeInitializer<transform_pymod_tag> instance;
    return instance;
}

void initialize_transform(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_FourierTransform(mod);
    };

    OneTimeInitializer<transform_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
