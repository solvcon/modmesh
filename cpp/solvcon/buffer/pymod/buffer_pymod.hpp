#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h> // Must be the first include.
#include <pybind11/stl.h>

#include <solvcon/python/common.hpp>

namespace solvcon
{

namespace python
{

void initialize_buffer(pybind11::module & mod);
void wrap_ConcreteBuffer(pybind11::module & mod);
void wrap_SimpleArray(pybind11::module & mod);
void wrap_SimpleArrayPlex(pybind11::module & mod);

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
