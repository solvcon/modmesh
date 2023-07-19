#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <modmesh/modmesh.hpp>
#include <modmesh/python/common.hpp>
#include <modmesh/io/io.hpp>

namespace modmesh
{

namespace python
{

void initialize_io(pybind11::module & mod);
void wrap_Gmsh(pybind11::module & mod);

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
