#include <gtest/gtest.h>
#include <thread>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

#define CALLPROFILER 1
#include <modmesh/toggle/callprofiler.hpp>
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

    RadixTree<CallerProfile> & radix_tree()
    {
        return pProfiler->m_radix_tree;
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

TEST_F(CallProfilerTest, test_print_result)
{
    pProfiler->reset();

    foo1();

    std::stringstream ss;
    pProfiler->print_profiling_result(ss);
}

static bool diff_time(std::chrono::nanoseconds raw_nano_time, int time_ms, int factor = 1)
{
    const int error = 5 * factor; // a function call can has error in about 5 ms on macOS
    return std::abs(raw_nano_time.count() / 1e6 - time_ms) < error;
}

TEST_F(CallProfilerTest, test_simple_case1)
{
    pProfiler->reset();

    foo1();

    // void modmesh::foo1() - Total Time: 61 ms, Call Count: 1
    //   void modmesh::foo2() - Total Time: 54 ms, Call Count: 1
    //      void modmesh::foo3() - Total Time: 19 ms, Call Count: 1

    int key = 0;

    auto * node1 = radix_tree().get_current_node()->get_child(key++);
    EXPECT_EQ(node1->data().caller_name, "void modmesh::foo1()");
    EXPECT_EQ(node1->data().call_count, 1);
    EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2 + uniqueTime3, 3));

    auto * node2 = node1->get_child(key++);
    EXPECT_EQ(node2->data().caller_name, "void modmesh::foo2()");
    EXPECT_EQ(node2->data().call_count, 1);
    EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1 + uniqueTime2, 2));

    auto * node3 = node2->get_child(key++);
    EXPECT_EQ(node3->data().caller_name, "void modmesh::foo3()");
    EXPECT_EQ(node3->data().call_count, 1);
    EXPECT_TRUE(diff_time(node3->data().total_time, uniqueTime1));
}

TEST_F(CallProfilerTest, simple_case_2)
{
    pProfiler->reset();

    foo1();
    foo2();
    foo3();
    foo3();
    // void modmesh::foo1 - Total Time: 61 ms, Call Count: 1
    //   void modmesh::foo2 - Total Time: 54 ms, Call Count: 1
    //     void modmesh::foo3 - Total Time: 19 ms, Call Count: 1
    // void modmesh::foo2 - Total Time: 54 ms, Call Count: 1
    //   void modmesh::foo3 - Total Time: 19 ms, Call Count: 1
    // void modmesh::foo3 - Total Time: 38 ms, Call Count: 2

    // for first `foo1()` call
    {
        auto name1 = "void modmesh::foo1()";
        auto * node1 = radix_tree().get_current_node()->get_child(name1);
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, name1);
        EXPECT_EQ(node1->data().call_count, 1);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2 + uniqueTime3, 3));

        auto name2 = "void modmesh::foo2()";
        auto * node2 = node1->get_child(name2);
        EXPECT_NE(node2, nullptr);
        EXPECT_EQ(node2->data().caller_name, name2);
        EXPECT_EQ(node2->data().call_count, 1);
        EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1 + uniqueTime2, 2));

        auto name3 = "void modmesh::foo3()";
        auto * node3 = node2->get_child(name3);
        EXPECT_NE(node3, nullptr);
        EXPECT_EQ(node3->data().caller_name, name3);
        EXPECT_EQ(node3->data().call_count, 1);
        EXPECT_TRUE(diff_time(node3->data().total_time, uniqueTime1));
    }

    // for  `foo2()` call
    {
        auto name1 = "void modmesh::foo2()";
        auto * node1 = radix_tree().get_current_node()->get_child(name1); // id = 1, because previously already assigned in the map, FIXME: probably find a better way than hard
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, name1);
        EXPECT_EQ(node1->data().call_count, 1);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2, 2));

        auto name2 = "void modmesh::foo3()";
        auto * node2 = node1->get_child(name2);
        EXPECT_NE(node2, nullptr);
        EXPECT_EQ(node2->data().caller_name, name2);
        EXPECT_EQ(node2->data().call_count, 1);
        EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1));
    }

    // for  two `foo3()` call
    {
        auto name1 = "void modmesh::foo3()";
        auto * node1 = radix_tree().get_current_node()->get_child(name1);
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, name1);
        EXPECT_EQ(node1->data().call_count, 2);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 * 2, 2));
    }
}

} // namespace modmesh
