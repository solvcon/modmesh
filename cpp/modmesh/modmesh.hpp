#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * \file This is a template library for the meshes for numerical calculations
 * of partial differential equations.
 */

#include <modmesh/base.hpp>
#include <modmesh/math/math.hpp>
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/grid.hpp>
#include <modmesh/mesh/mesh.hpp>
#include <modmesh/toggle/toggle.hpp>
#include <modmesh/transform/transform.hpp>

// TODO Add MSVC case once sanitizer can be default turned on for CI testing
#if defined(USE_SANITIZER) && (defined(__clang__) || defined(__GNUC__))
#define ASAN_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
#define ASAN_NO_SANITIZE_ADDRESS
#endif

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
