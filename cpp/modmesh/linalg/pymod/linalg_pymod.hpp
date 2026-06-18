#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h> // Must be the first include.
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <modmesh/python/common.hpp>
#include <modmesh/linalg/linalg.hpp>

namespace modmesh
{

namespace python
{

void initialize_linalg(pybind11::module & mod);
void wrap_factorization(pybind11::module & mod);
void wrap_states_info(pybind11::module & mod);
void wrap_kalman_filter(pybind11::module & mod);
void wrap_EigenSystem(pybind11::module & mod);
void wrap_LuFactorization(pybind11::module & mod);

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
