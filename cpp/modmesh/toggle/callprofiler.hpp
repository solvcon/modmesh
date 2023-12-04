#pragma once
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

    void start_time_watch()
    {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    void stop_time_watch()
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
        callProfile.start_time_watch();
    }

    // Called when a function ends
    void end_caller(const std::string & caller_name)
    {

        CallerProfile & callProfile = m_radix_tree.get_current_node()->data();

        // Update profiling information
        callProfile.stop_time_watch();
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
                profile.stop_time_watch();
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
            Cancel();
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

    void Cancel()
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
