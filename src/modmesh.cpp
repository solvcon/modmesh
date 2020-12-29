/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD-style license; see COPYING
 */

#include <modmesh/python/python.hpp> // Must be the first include.
#include <modmesh/modmesh.hpp>

PYBIND11_MODULE(_modmesh, mod) // NOLINT
{
    modmesh::python::initialize(mod);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
