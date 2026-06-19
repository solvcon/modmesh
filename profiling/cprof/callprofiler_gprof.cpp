/*
 * Copyright (c) 2026, modmesh contributors
 * BSD-style license; see COPYING
 */

#define MODMESH_PROFILE 1
#include <callprofiler_workload.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <utility>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace profiling
{

namespace workload = modmesh::profiling;

using clock_type = std::chrono::steady_clock;
using profiler_type = modmesh::CallProfiler;
using runner_type = void (*)(std::size_t);

struct case_definition
{
    std::string_view m_label;
    runner_type m_runner;
};

std::array<case_definition, 4> const case_definitions{{
    {"wide_siblings", &workload::run_wide_siblings},
    {"deep_chain", &workload::run_deep_chain},
    {"balanced_tree", &workload::run_balanced_tree},
    {"hot_name_reuse", &workload::run_hot_name_reuse},
}};

void configure_large_stack()
{
#if defined(__linux__)
    rlimit limit{};
    if (getrlimit(RLIMIT_STACK, &limit) == 0)
    {
        if (RLIM_INFINITY == limit.rlim_max || limit.rlim_cur < limit.rlim_max)
        {
            limit.rlim_cur = limit.rlim_max;
            static_cast<void>(setrlimit(RLIMIT_STACK, &limit));
        }
    }
#endif
}

template <typename Runner>
void run_case(std::string_view label, std::size_t operation_count, std::size_t repeat_count, Runner && runner)
{
    profiler_type & profiler = profiler_type::instance();
    std::chrono::duration<double> elapsed{0.0};

    for (std::size_t repeat = 0; repeat < repeat_count; ++repeat)
    {
        profiler.reset();

        auto const start_time = clock_type::now();
        std::forward<Runner>(runner)();
        auto const stop_time = clock_type::now();

        elapsed += stop_time - start_time;
    }

    std::cout << "RESULT workload=" << label
              << " operations=" << operation_count
              << " repeats=" << repeat_count
              << " workload_seconds=" << elapsed.count()
              << '\n';

    profiler.reset();
}

std::size_t parse_size(char const * value)
{
    return static_cast<std::size_t>(std::strtoull(value, nullptr, 10));
}

case_definition const * find_case(std::string_view label)
{
    for (case_definition const & definition : case_definitions)
    {
        if (definition.m_label == label)
        {
            return &definition;
        }
    }
    return nullptr;
}

bool run_named_case(std::string_view label, std::size_t size, std::size_t repeat_count)
{
    case_definition const * definition = find_case(label);
    if (definition == nullptr)
    {
        return false;
    }

    run_case(definition->m_label, size, repeat_count, [definition, size]()
             { definition->m_runner(size); });
    return true;
}

} /* namespace profiling */

int main(int argc, char ** argv)
{
    if (argc == 4)
    {
        profiling::configure_large_stack();
        bool const completed = profiling::run_named_case(
            argv[1],
            profiling::parse_size(argv[2]),
            profiling::parse_size(argv[3]));
        return completed ? 0 : 2;
    }

    return 2;
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
