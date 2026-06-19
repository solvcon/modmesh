#pragma once

/*
 * Copyright (c) 2026, modmesh contributors
 * BSD-style license; see COPYING
 */

#ifndef MODMESH_PROFILE
#define MODMESH_PROFILE 1
#endif

#include <modmesh/toggle/RadixTree.hpp>

#include <cstddef>

#if defined(__clang__) || defined(__GNUC__)
#define MODMESH_CPROF_NOINLINE __attribute__((noinline))
#define MODMESH_CPROF_NOINST __attribute__((no_instrument_function))
#else
#define MODMESH_CPROF_NOINLINE
#define MODMESH_CPROF_NOINST
#endif

namespace modmesh::profiling
{

void run_wide_siblings(std::size_t size);
void run_deep_chain(std::size_t size);
void run_balanced_tree(std::size_t size);
void run_hot_name_reuse(std::size_t size);

namespace detail
{

enum class WorkloadShape
{
    flat,
    list,
    tree,
};

using profile_function_type = void (*)(std::size_t, std::size_t);

extern WorkloadShape active_shape;
extern std::size_t active_size;

void call_profile_function(std::size_t index, std::size_t begin, std::size_t end);

} /* namespace detail */

} /* namespace modmesh::profiling */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
