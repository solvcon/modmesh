/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "modmesh/modmesh.hpp"
#include "modmesh/python/python.hpp"

namespace modmesh
{

namespace python
{

void initialize(pybind11::module & mod)
{

    WrapGrid1d::commit(mod);
    WrapGrid2d::commit(mod);
    WrapGrid3d::commit(mod);

}

} /* end namespace python */

} /* end namespace modmesh */

PYBIND11_MODULE(_modmesh, mod)
{
    modmesh::python::initialize(mod);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
