#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * \file This is a template library for the meshes for numerical calculations
 * of partial differential equations.
 */

#include <solvcon/base.hpp>
#include <solvcon/math/math.hpp>
#include <solvcon/buffer/buffer.hpp>
#include <solvcon/grid.hpp>
#include <solvcon/mesh/mesh.hpp>
#include <solvcon/toggle/toggle.hpp>
#include <solvcon/transform/transform.hpp>

// TODO Add MSVC case once sanitizer can be default turned on for CI testing
#if defined(USE_SANITIZER) && (defined(__clang__) || defined(__GNUC__))
#define ASAN_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
#define ASAN_NO_SANITIZE_ADDRESS
#endif

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
