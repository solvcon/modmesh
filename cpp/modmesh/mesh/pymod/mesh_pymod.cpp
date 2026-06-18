/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/mesh/pymod/mesh_pymod.hpp> // Must be the first include.

namespace modmesh
{

namespace python
{

struct mesh_pymod_tag;

template <>
OneTimeInitializer<mesh_pymod_tag> & OneTimeInitializer<mesh_pymod_tag>::me()
{
    static OneTimeInitializer<mesh_pymod_tag> instance;
    return instance;
}

void initialize_mesh(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_StaticGrid(mod);
        wrap_StaticMesh(mod);
    };

    OneTimeInitializer<mesh_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
