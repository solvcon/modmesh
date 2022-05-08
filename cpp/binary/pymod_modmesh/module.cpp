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
 * ARE DISCLAIMED.
 */

#include <modmesh/python/python.hpp> // Must be the first include.
#include <modmesh/python/wrapper/modmesh/modmesh.hpp>

#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace modmesh
{

namespace python
{

namespace detail
{

static void import_numpy()
{
    auto local_import_numpy = []()
    {
        import_array2("cannot import numpy", false); // or numpy c api segfault.
        return true;
    };
    if (!local_import_numpy())
    {
        throw pybind11::error_already_set();
    }
}

} /* end namespace detail */

struct modmesh_pymod_tag;

static void initialize_modmesh(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        detail::import_numpy();

        wrap_profile(mod);
        wrap_ConcreteBuffer(mod);
        wrap_SimpleArray(mod);
        wrap_StaticGrid(mod);
        wrap_StaticMesh(mod);
    };

    OneTimeInitializer<modmesh_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

PYBIND11_MODULE(_modmesh, mod) // NOLINT
{
    modmesh::python::initialize_modmesh(mod);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
