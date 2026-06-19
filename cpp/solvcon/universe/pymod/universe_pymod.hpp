#pragma once

/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h> // Must be the first include.
#include <pybind11/stl.h>

#include <solvcon/python/common.hpp>
#include <solvcon/universe/universe.hpp>

namespace solvcon
{

namespace python
{

void initialize_universe(pybind11::module & mod);
void wrap_shape0d(pybind11::module & mod);
void wrap_shape1d(pybind11::module & mod);
void wrap_shape2d(pybind11::module & mod);
void wrap_shape3d(pybind11::module & mod);
void wrap_view_transform2d(pybind11::module & mod);
void wrap_World(pybind11::module & mod);

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: