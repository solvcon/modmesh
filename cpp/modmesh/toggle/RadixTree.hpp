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

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <stack>
#include <unordered_map>

namespace modmesh
{
/**
 * Simple Timed Entry
 */
class TimedEntry
{
public:
    size_t count() const { return m_count; }
    double time() const { return m_time; }

    TimedEntry & add_time(double time)
    {
        ++m_count;
        m_time += time;
        return *this;
    }

    friend std::ostream & operator<<(std::ostream & os, const TimedEntry & entry)
    {
        os << "Count: " << entry.count() << " - Time: " << entry.time();
        return os;
    }

private:
    size_t m_count = 0;
    double m_time = 0.0;
}; /* end class TimedEntry */

template <typename T>
class RadixTreeNode
{
public:

    using child_list_type = std::list<std::unique_ptr<RadixTreeNode<T>>>;
    using key_type = int32_t;

    static_assert(std::is_integral_v<key_type> && std::is_signed_v<key_type>, "signed integral required");

    RadixTreeNode(std::string const & name, key_type key)
        : m_name(name)
        , m_key(key)
        , m_prev(nullptr)
    {
    }

    RadixTreeNode() = default;
    RadixTreeNode(RadixTreeNode const &) = default;
    RadixTreeNode(RadixTreeNode &&) = default;
    RadixTreeNode & operator=(RadixTreeNode const &) = default;
    RadixTreeNode & operator=(RadixTreeNode &&) = default;
    ~RadixTreeNode() = default;

    key_type key() const { return m_key; }
    const std::string & name() const { return m_name; }
    T & data() { return m_data; }
    const T & data() const { return m_data; }
    const child_list_type & children() const { return m_children; }

    RadixTreeNode<T> * add_child(std::string const & name, key_type key)
    {
        auto new_child = std::make_unique<RadixTreeNode<T>>(name, key);
        new_child->m_prev = this;
        m_children.push_back(std::move(new_child));
        return m_children.back().get();
    }

    RadixTreeNode<T> * get_child(key_type key) const
    {
        auto it = std::find_if(m_children.begin(), m_children.end(), [&](const auto & child)
                               { return child->key() == key; });
        return (it != m_children.end()) ? it->get() : nullptr;
    }

    RadixTreeNode<T> * get_child(std::string name) const
    {
        auto it = std::find_if(m_children.begin(), m_children.end(), [&](const auto & child)
                               { return child->name() == name; });
        return (it != m_children.end()) ? it->get() : nullptr;
    }

    RadixTreeNode<T> * get_prev() const { return m_prev; }

private:
    key_type m_key = -1;
    std::string m_name;
    T m_data;
    child_list_type m_children;
    RadixTreeNode<T> * m_prev = nullptr;
}; /* end class RadixTreeNode */

/*
Ref:
https://kalkicode.com/radix-tree-implementation
https://www.algotree.org/algorithms/trie/
*/
template <typename T>
class RadixTree
{
public:
    using key_type = typename RadixTreeNode<T>::key_type;
    RadixTree()
        : m_root(std::make_unique<RadixTreeNode<T>>())
        , m_current_node(m_root.get())
    {
    }

    T & entry(const std::string & name)
    {
        key_type id = get_id(name);

        RadixTreeNode<T> * child = m_current_node->get_child(id);

        if (!child)
        {
            m_current_node = m_current_node->add_child(name, id);
        }
        else
        {
            m_current_node = child;
        }

        return m_current_node->data();
    }

    void move_current_to_parent()
    {
        if (m_current_node != m_root.get())
        {
            m_current_node = m_current_node->get_prev();
        }
    }

    void reset()
    {
        m_root = std::move(std::make_unique<RadixTreeNode<T>>());
        m_current_node = m_root.get();
        m_id_map.clear();
        m_unique_id = 0;
    }

    bool is_root() const
    {
        return m_current_node == m_root.get();
    }

    RadixTreeNode<T> * get_current_node() const { return m_current_node; }
    key_type get_unique_node() const { return m_unique_id; }

private:
    key_type get_id(const std::string & name)
    {
        auto [it, inserted] = m_id_map.try_emplace(name, m_unique_id++);
        return it->second;
    }

    // FIXME: Use raw pointer for m_root.
    std::unique_ptr<RadixTreeNode<T>> m_root;
    RadixTreeNode<T> * m_current_node;
    std::unordered_map<std::string, key_type> m_id_map;
    key_type m_unique_id = 0;
}; /* end class RadixTree */

// The profiling result of the caller
struct CallerProfile
{
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
        call_count++;
    }

    std::chrono::high_resolution_clock::time_point start_time;
    std::function<void()> cancel_callback;
    std::string caller_name;
    std::chrono::nanoseconds total_time = std::chrono::nanoseconds(0); /// use nanoseconds to have higher precision
    int call_count = 0;
    bool is_running = false;
}; /* end struct CallerProfile */

namespace detail
{
class CallProfilerTest; // for gtest
}

/// The profiler that profiles the hierarchical caller stack.
class CallProfiler
{
private:
    CallProfiler() = default;

public:
    /// A singleton.
    static CallProfiler & instance()
    {
        static CallProfiler instance;
        return instance;
    }

    CallProfiler(CallProfiler const &) = delete;
    CallProfiler(CallProfiler &&) = delete;
    CallProfiler & operator=(CallProfiler const &) = delete;
    CallProfiler & operator=(CallProfiler &&) = delete;
    ~CallProfiler() = default;

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
        CallerProfile & call_profile = m_radix_tree.get_current_node()->data();
        call_profile.stop_stopwatch(); // Update profiling information
        m_radix_tree.move_current_to_parent(); // Pop the caller from the call stack
    }

    /// Print the profiling information
    void print_profiling_result(std::ostream & outstream) const
    {
        print_profiling_result(*(m_radix_tree.get_current_node()), 0, outstream);
    }

    /// Reset the profiler
    void reset();

private:
    void print_profiling_result(const RadixTreeNode<CallerProfile> & node, const int depth, std::ostream & outstream) const;

private:
    RadixTree<CallerProfile> m_radix_tree; /// the data structure of the callers

    friend detail::CallProfilerTest;
}; /* end class CallProfiler */

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

    CallProfilerProbe(CallProfilerProbe const &) = delete;
    CallProfilerProbe(CallProfilerProbe &&) = delete;
    CallProfilerProbe & operator=(CallProfilerProbe const &) = delete;
    CallProfilerProbe & operator=(CallProfilerProbe &&) = delete;

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
}; /* end struct CallProfilerProbe */

#ifdef CALLPROFILER

#ifdef _MSC_VER
// ref: https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros
#define __CROSS_PRETTY_FUNCTION__ __FUNCSIG__
#else
// ref: https://gcc.gnu.org/onlinedocs/gcc/Function-Names.html
#define __CROSS_PRETTY_FUNCTION__ __PRETTY_FUNCTION__
#endif
#define USE_CALLPROFILER_PROFILE_THIS_FUNCTION() modmesh::CallProfilerProbe __profilerProbe##__COUNTER__(modmesh::CallProfiler::instance(), __CROSS_PRETTY_FUNCTION__)
#define USE_CALLPROFILER_PROFILE_THIS_SCOPE(scopeName) modmesh::CallProfilerProbe __profilerProbe##__COUNTER__(modmesh::CallProfiler::instance(), scopeName)
#else
#define USE_CALLPROFILER_PROFILE_THIS_FUNCTION() // do nothing
#define USE_CALLPROFILER_PROFILE_THIS_SCOPE(scopeName) // do nothing
#endif

} /* end namespace modmesh */
// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
