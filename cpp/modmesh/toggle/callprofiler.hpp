#pragma once

/*
 * Copyright (c) 2023, Quentin Tsai <quentin.tsai.tw@gmail.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <functional>
#include <modmesh/toggle/RadixTree.hpp>
#include <ostream>
#include <stack>
#include <unordered_map>

namespace modmesh
{

// The profiling result of the caller
struct CallerProfile
{
    CallerProfile()
        : total_time(0)
    {
    }

    void start_stopwatch()
    {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    void stop_stopwatch()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        total_time += elapsed_time;
    }

    std::chrono::high_resolution_clock::time_point start_time;
    std::function<void()> cancel_callback;
    std::string caller_name;
    std::chrono::nanoseconds total_time;
    int callCount = 0;
    bool is_running = false;
};

/// The profiler that profiles the hierarchical caller stack.
class CallProfiler
{
public:
    /// A singleton.
    static CallProfiler & instance()
    {
        static CallProfiler instance;
        return instance;
    }

    // Called when a function starts
    void start_caller(const std::string & caller_name, std::function<void()> cancel_callback)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        m_radix_tree.entry(caller_name);
        CallerProfile & callProfile = m_radix_tree.get_current_node()->data();
        callProfile.caller_name = caller_name;
        callProfile.start_stopwatch();
    }

    // Called when a function ends
    void end_caller(const std::string & caller_name)
    {

        CallerProfile & callProfile = m_radix_tree.get_current_node()->data();

        // Update profiling information
        callProfile.stop_stopwatch();
        callProfile.callCount++;

        // Pop the caller from the call stack
        m_radix_tree.move_current_to_parent();
    }

    /// Print the profiling information
    void print_profiling_result(std::ostream & outstream) const
    {
        _print_profiling_result(*(m_radix_tree.get_current_node()), 0, outstream);
    }

    /// Result the profiler
    void reset()
    {
        RadixTreeNode<CallerProfile> * newNode;
        RadixTreeNode<CallerProfile> * currentNode = m_radix_tree.get_current_node();

        while (!m_radix_tree.is_root())
        {
            CallerProfile & profile = m_radix_tree.get_current_node()->data();
            if (profile.is_running)
            {
                profile.stop_stopwatch();
            }
            m_radix_tree.move_current_to_parent();
        }
        m_radix_tree.reset();
    }

private:
    CallProfiler() = default;

    void _print_profiling_result(const RadixTreeNode<CallerProfile> & node, const int depth, std::ostream & outstream) const
    {
        for (int i = 0; i < depth; ++i)
        {
            outstream << "  ";
        }

        auto profile = node.data();

        if (depth == 0)
        {
            outstream << "Profiling Result" << std::endl;
        }
        else
        {
            outstream << profile.caller_name << " - Total Time: " << profile.total_time.count() / 1000 << " ms, Call Count: " << profile.callCount << std::endl;
        }

        for (const auto & child : node.children())
        {
            _print_profiling_result(*child, depth + 1, outstream);
        }
    }

private:
    RadixTree<CallerProfile> m_radix_tree; /// the data structure of the callers
};

/// Utility to profile a call
class CallProfilerProbe
{
public:
    CallProfilerProbe(CallProfiler & profiler, const char * caller_name)
        : m_profiler(profiler)
        , m_caller_name(caller_name)
    {
        auto cancel_callback = [&]()
        {
            cancel();
        };
        m_profiler.start_caller(m_caller_name, cancel_callback);
    }

    ~CallProfilerProbe()
    {
        if (!m_cancel)
        {
            m_profiler.end_caller(m_caller_name);
        }
    }

    void cancel()
    {
        m_cancel = true;
    }

private:
    const char * m_caller_name;
    bool m_cancel = false;
    CallProfiler & m_profiler;
};

#ifdef CALLPROFILER
#define USE_CALLPROFILER_PROFILE_THIS_FUNCTION() modmesh::CallProfilerProbe __profilerProbe##__COUNTER__(modmesh::CallProfiler::instance(), __PRETTY_FUNCTION__)
#define USE_CALLPROFILER_PROFILE_THIS_SCOPE(scopeName) modmesh::CallProfilerProbe __profilerProbe##__COUNTER__(modmesh::CallProfiler::instance(), scopeName)
#else
#define USE_CALLPROFILER_PROFILE_THIS_FUNCTION() // do nothing
#define USE_CALLPROFILER_PROFILE_THIS_SCOPE(scopeName) // do nothing
#endif
} // namespace modmesh
