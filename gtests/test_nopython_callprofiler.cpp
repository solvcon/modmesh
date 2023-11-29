#include <modmesh/toggle/callprofiler.hpp>
#include <gtest/gtest.h>
#include <thread>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(CallProfiler, construction)
{
    namespace mm = modmesh;
    mm::CallProfiler::instance();
}

constexpr int uniqueTime1 = 19;
constexpr int uniqueTime2 = 35;
constexpr int uniqueTime3 = 7;

void foo3()
{
    modmesh::USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime1));
}

void foo2()
{
    modmesh::USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime2));
    foo3();
}

void foo1()
{
    modmesh::USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    foo2();
    std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime3));
}

TEST(CallProfilerCase1, construction)
{
    namespace mm = modmesh;

    mm::CallProfiler& profiler = mm::CallProfiler::instance();
    profiler.reset();

    foo1();

    std::stringstream ss;
    profiler.print_profiling_result(ss);

    const char * answer = R"(Profiling Result
  foo1 - Total Time: 61 ms, Call Count: 1
    foo2 - Total Time: 54 ms, Call Count: 1
      foo3 - Total Time: 19 ms, Call Count: 1
)";

    EXPECT_EQ(ss.str(), answer);
}

TEST(CallProfilerCase2, construction)
{
    namespace mm = modmesh;

    mm::CallProfiler& profiler = mm::CallProfiler::instance();
    profiler.reset();

    foo1();
    foo2();
    foo3();
    foo3();

    std::stringstream ss;
    profiler.print_profiling_result(ss);

    const char * answer = R"(Profiling Result
  foo1 - Total Time: 61 ms, Call Count: 1
    foo2 - Total Time: 115 ms, Call Count: 1
      foo3 - Total Time: 73 ms, Call Count: 1
  foo2 - Total Time: 73 ms, Call Count: 1
    foo3 - Total Time: 19 ms, Call Count: 1
  foo3 - Total Time: 38 ms, Call Count: 2
)";

    EXPECT_EQ(ss.str(), answer);
}