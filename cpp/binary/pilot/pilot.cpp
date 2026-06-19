/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/python/python.hpp> // Must be the first include.
#include <solvcon/python/module.hpp>

PYBIND11_EMBEDDED_MODULE(_solvcon, mod) // NOLINT
{
    solvcon::python::initialize(mod);
}

int main(int argc, char ** argv)
{
    return solvcon::python::program_entrance(argc, argv);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: