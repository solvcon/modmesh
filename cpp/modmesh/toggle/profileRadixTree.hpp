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

#include <modmesh/toggle/RadixTree.hpp>
#include <modmesh/toggle/profile.hpp>
#include <thread>

namespace modmesh
{

class TimeRegistryRadixTree
{

public:

    /// The singleton.
    static TimeRegistryRadixTree & me()
    {
        static TimeRegistryRadixTree inst;
        return inst;
    }

    void add(std::string const & name)
    {
        m_tree.entry(name);
    }

    void add_time(double time)
    {
        m_tree.get_current_node()->data().add_time(time);
        m_tree.move_current_to_parent();
    }

    void add(const char * name)
    {
        add(std::string(name));
    }

    void reset() { m_tree.clear(); }

    void report()
    {
        m_tree.printTree<TimedEntry>();
    }

    TimeRegistryRadixTree(TimeRegistryRadixTree const &) = delete;
    TimeRegistryRadixTree(TimeRegistryRadixTree &&) = delete;
    TimeRegistryRadixTree & operator=(TimeRegistryRadixTree const &) = delete;
    TimeRegistryRadixTree & operator=(TimeRegistryRadixTree &&) = delete;

    ~TimeRegistryRadixTree() // NOLINT(modernize-use-equals-default)
    {
        // Uncomment for debugging.
        // std::cout << report();
    }

private:

    TimeRegistryRadixTree() = default;

    RadixTree<TimedEntry> m_tree;

}; /* end struct TimeRegistryRadixTree */

class ProfilerProbe
{

public:

    ProfilerProbe() = delete;
    ProfilerProbe(ProfilerProbe const &) = delete;
    ProfilerProbe(ProfilerProbe &&) = delete;
    ProfilerProbe & operator=(ProfilerProbe const &) = delete;
    ProfilerProbe & operator=(ProfilerProbe &&) = delete;

    explicit ProfilerProbe(const char * name)
        : m_name(name)
    {
        TimeRegistryRadixTree::me().add(m_name);
    }

    ~ProfilerProbe()
    {
        TimeRegistryRadixTree::me().add_time(m_sw.lap());
    }

private:

    StopWatch m_sw;
    char const * m_name;

}; /* end class ProfilerProbe */

} /* end namespace modmesh */

// ref: https://gcc.gnu.org/onlinedocs/gcc/Function-Names.html
#define MODMESH_TIME_RADIX() \
    ProfilerProbe _local_scoped_timer_##__LINE__(__PRETTY_FUNCTION__);

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
