#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <solvcon/solvcon.hpp>
#include <solvcon/python/common.hpp>
#include <solvcon/inout/inout.hpp>

namespace solvcon
{

namespace python
{

void initialize_inout(pybind11::module & mod);
void wrap_Gmsh(pybind11::module & mod);
void wrap_Plot3d(pybind11::module & mod);

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
