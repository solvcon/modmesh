/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/python/python.hpp> // Must be the first include.
#include <solvcon/python/module.hpp>

PYBIND11_MODULE(_solvcon, mod) // NOLINT
{
    solvcon::python::initialize(mod);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
