#pragma once
#include <chrono>
#include <ostream>
#include <stack>
#include <unordered_map>
#include "RadixTree.hpp"

namespace modmesh
{

// Define a structure for each node in the radix tree
struct CallerProfile
{
    std::string functionName;
    std::chrono::milliseconds totalTime;
    int callCount = 0;
};

// Define a structure to represent a function call
struct CallerInfo
{
    std::string callerName;
    std::chrono::high_resolution_clock::time_point startTime;
};

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
    void start_caller(const std::string & callerName)
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        m_callStack.push({callerName, startTime});
        m_radixTree.entry(callerName);
    }

    // Called when a function ends
    void end_caller()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_callStack.top().startTime);

        // Update the radix tree with profiling information
        auto & callerName = m_callStack.top().callerName;

        CallerProfile & callProfile = m_radixTree.get_current_node()->data();

        // Update profiling information
        callProfile.totalTime += elapsedTime;
        callProfile.callCount++;
        callProfile.functionName = callerName;

        // Pop the function from the call stack
        m_callStack.pop();
        m_radixTree.move_current_to_parent();
    }

    /// Print the profiling information
    void print_profiling_result(std::ostream & outstream) const
    {
        _print_profiling_result(*(m_radixTree.get_current_node()), 0, outstream);
    }

    void reset()
    {
        while (!m_callStack.empty())
        {
            m_callStack.pop();
        }
        m_radixTree.reset();
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
            outstream << profile.functionName << " - Total Time: " << profile.totalTime.count() << " ms, Call Count: " << profile.callCount << std::endl;
        }

        for (const auto & child : node.children())
        {
            _print_profiling_result(*child, depth + 1, outstream);
        }
    }

private:
    std::stack<CallerInfo> m_callStack;
    RadixTree<CallerProfile> m_radixTree;
};

/// Utility to time how long a scope takes
class CallProfilerProbe
{
public:
    CallProfilerProbe(CallProfiler & profiler, const char * callerName)
        : m_profiler(profiler)
    {
        m_profiler.start_caller(callerName);
    }

    ~CallProfilerProbe()
    {
        m_profiler.end_caller();
    }

private:
    CallProfiler & m_profiler;
};

#define USE_CALLPROFILER_PROFILE_THIS_FUNCTION() CallProfilerProbe __profilerProbe##__COUNTER__(modmesh::CallProfiler::instance(), __FUNCTION__)
#define USE_CALLPROFILER_PROFILE_THIS_SCOPE(scopeName) CallProfilerProbe __profilerProbe##__COUNTER__(modmesh::CallProfiler::instance(), scopeName)

} // namespace modmesh
