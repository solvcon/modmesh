#include <gtest/gtest.h>
#include <thread>

#include <modmesh/toggle/callprofiler.hpp>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace modmesh
{
class CallProfilerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CallProfiler & profiler = CallProfiler::instance();
        pProfiler = &profiler;
    }

    std::stack<CallerInfo> & call_stack()
    {
        return pProfiler->m_callStack;
    }

    CallProfiler * pProfiler;
};

constexpr int uniqueTime1 = 19;
constexpr int uniqueTime2 = 35;
constexpr int uniqueTime3 = 7;

void foo3()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime1));
}

void foo2()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime2));
    foo3();
}

void foo1()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    foo2();
    std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime3));
}

TEST_F(CallProfilerTest, test_reset)
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    pProfiler->reset();
    EXPECT_EQ(call_stack().empty(), true);
}

TEST_F(CallProfilerTest, simple_case_1)
{
    pProfiler->reset();

    foo1();

    std::stringstream ss;
    pProfiler->print_profiling_result(ss);

    const char * answer = R"(Profiling Result
  foo1 - Total Time: 61 ms, Call Count: 1
    foo2 - Total Time: 54 ms, Call Count: 1
      foo3 - Total Time: 19 ms, Call Count: 1
)";

    EXPECT_EQ(ss.str(), answer);
}

TEST_F(CallProfilerTest, simple_case_2)
{
    pProfiler->reset();

    foo1();
    foo2();
    foo3();
    foo3();

    std::stringstream ss;
    pProfiler->print_profiling_result(ss);

    const char * answer = R"(Profiling Result
  foo1 - Total Time: 61 ms, Call Count: 1
    foo2 - Total Time: 54 ms, Call Count: 1
      foo3 - Total Time: 19 ms, Call Count: 1
  foo2 - Total Time: 54 ms, Call Count: 1
    foo3 - Total Time: 19 ms, Call Count: 1
  foo3 - Total Time: 38 ms, Call Count: 2
)";

    EXPECT_EQ(ss.str(), answer);
}

} // namespace modmesh
