#include <gtest/gtest.h>
#include <thread>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

#define CALLPROFILER 1
#include <modmesh/toggle/RadixTree.hpp>
namespace modmesh
{

namespace detail
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
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime1)
    {
        // use busy loop to get a precise duration
    }
}

void foo2()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime2)
    {
        // use busy loop to get a precise duration
    }
    foo3();
}

void foo1()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    foo2();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime3)
    {
        // use busy loop to get a precise duration
    }
}

TEST_F(CallProfilerTest, test_print_result)
{
    pProfiler->reset();

    foo1();

    std::stringstream ss;
    pProfiler->print_profiling_result(ss);
}

static bool diff_time(std::chrono::nanoseconds raw_nano_time, int time_ms)
{
    constexpr int error = 5; // a reasonable error
    return std::abs(raw_nano_time.count() / 1e6 - time_ms) < error;
}

#ifdef _MSC_VER
auto foo1Name = "void __cdecl modmesh::detail::foo1(void)";
#else
auto foo1Name = "void modmesh::detail::foo1()";
#endif

#ifdef _MSC_VER
auto foo2Name = "void __cdecl modmesh::detail::foo2(void)";
#else
auto foo2Name = "void modmesh::detail::foo2()";
#endif

#ifdef _MSC_VER
auto foo3Name = "void __cdecl modmesh::detail::foo3(void)";
#else
auto foo3Name = "void modmesh::detail::foo3()";
#endif

TEST_F(CallProfilerTest, test_simple_case1)
{
    pProfiler->reset();

    foo1();

    // Example:
    // void modmesh::foo1() - Total Time: 61 ms, Call Count: 1
    //   void modmesh::foo2() - Total Time: 54 ms, Call Count: 1
    //      void modmesh::foo3() - Total Time: 19 ms, Call Count: 1

    int key = 0;

    auto * node1 = radix_tree().get_current_node()->get_child(key++);
    EXPECT_EQ(node1->data().caller_name, foo1Name);
    EXPECT_EQ(node1->data().call_count, 1);
    EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2 + uniqueTime3));

    auto * node2 = node1->get_child(key++);
    EXPECT_EQ(node2->data().caller_name, foo2Name);
    EXPECT_EQ(node2->data().call_count, 1);
    EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1 + uniqueTime2));

    auto * node3 = node2->get_child(key++);
    EXPECT_EQ(node3->data().caller_name, foo3Name);
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

    // Example:
    // void modmesh::foo1 - Total Time: 61 ms, Call Count: 1
    //   void modmesh::foo2 - Total Time: 54 ms, Call Count: 1
    //     void modmesh::foo3 - Total Time: 19 ms, Call Count: 1
    // void modmesh::foo2 - Total Time: 54 ms, Call Count: 1
    //   void modmesh::foo3 - Total Time: 19 ms, Call Count: 1
    // void modmesh::foo3 - Total Time: 38 ms, Call Count: 2

    // for first `foo1()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo1Name);
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo1Name);
        EXPECT_EQ(node1->data().call_count, 1);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2 + uniqueTime3));

        auto * node2 = node1->get_child(foo2Name);
        EXPECT_NE(node2, nullptr);
        EXPECT_EQ(node2->data().caller_name, foo2Name);
        EXPECT_EQ(node2->data().call_count, 1);
        EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1 + uniqueTime2));

        auto * node3 = node2->get_child(foo3Name);
        EXPECT_NE(node3, nullptr);
        EXPECT_EQ(node3->data().caller_name, foo3Name);
        EXPECT_EQ(node3->data().call_count, 1);
        EXPECT_TRUE(diff_time(node3->data().total_time, uniqueTime1));
    }

    // for  `foo2()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo2Name); // id = 1, because previously already assigned in the map, FIXME: probably find a better way than hard
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo2Name);
        EXPECT_EQ(node1->data().call_count, 1);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 + uniqueTime2));

        auto * node2 = node1->get_child(foo3Name);
        EXPECT_NE(node2, nullptr);
        EXPECT_EQ(node2->data().caller_name, foo3Name);
        EXPECT_EQ(node2->data().call_count, 1);
        EXPECT_TRUE(diff_time(node2->data().total_time, uniqueTime1));
    }

    // for  two `foo3()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo3Name);
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo3Name);
        EXPECT_EQ(node1->data().call_count, 2);
        EXPECT_TRUE(diff_time(node1->data().total_time, uniqueTime1 * 2));
    }
}

TEST_F(CallProfilerTest, cancel)
{
    pProfiler->reset();

    auto test1 = [&]()
    {
        USE_CALLPROFILER_PROFILE_THIS_FUNCTION();

        auto test2 = [&]()
        {
            USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
            pProfiler->cancel();
        };

        test2();
    };
    test1();

    EXPECT_EQ(radix_tree().get_unique_node(), 0);
}

#ifdef _MSC_VER
std::string serializeExample = R"x({"radix_tree": {"nodes": [{"key": -1,"name": "","data": {"start_time": 0,"caller_name": "","total_time": 0,"call_count": 0,"is_running": 0},"children": [0]},{"key": 0,"name": "void __cdecl modmesh::detail::foo1(void)","data": {"start_time": 18348582416327166,"caller_name": "void __cdecl modmesh::detail::foo1(void)","total_time": 61001042,"call_count": 1,"is_running": 1},"children": [1]},{"key": 1,"name": "void __cdecl modmesh::detail::foo2(void)","data": {"start_time": 18348582416327416,"caller_name": "void __cdecl modmesh::detail::foo2(void)","total_time": 54000667,"call_count": 1,"is_running": 1},"children": [2]},{"key": 2,"name": "void __cdecl modmesh::detail::foo3(void)","data": {"start_time": 18348582451327708,"caller_name": "void __cdecl modmesh::detail::foo3(void)","total_time": 19000167,"call_count": 1,"is_running": 1},"children": []}],"current_node": -1,"unique_id": 3}})x";
#else
std::string serializeExample = R"x({"radix_tree": {"nodes": [{"key": -1,"name": "","data": {"start_time": 0,"caller_name": "","total_time": 0,"call_count": 0,"is_running": 0},"children": [0]},{"key": 0,"name": "void modmesh::detail::foo1()","data": {"start_time": 18348582416327166,"caller_name": "void modmesh::detail::foo1()","total_time": 61001042,"call_count": 1,"is_running": 1},"children": [1]},{"key": 1,"name": "void modmesh::detail::foo2()","data": {"start_time": 18348582416327416,"caller_name": "void modmesh::detail::foo2()","total_time": 54000667,"call_count": 1,"is_running": 1},"children": [2]},{"key": 2,"name": "void modmesh::detail::foo3()","data": {"start_time": 18348582451327708,"caller_name": "void modmesh::detail::foo3()","total_time": 19000167,"call_count": 1,"is_running": 1},"children": []}],"current_node": -1,"unique_id": 3}})x";
#endif

std::string start_time_str = R"("start_time": )";
std::string total_time_str = R"("total_time": )";
std::string caller_name_str = R"(,"caller_name": )";
std::string call_count_str = R"(,"call_count": )";

TEST_F(CallProfilerTest, test_serialization)
{
    pProfiler->reset();

    foo1();

    // Example:
    //  {
    //      "radix_tree":
    //      {
    //          "nodes":[
    //              {"key":-1,"name":"","data":{"start_time": 0,"caller_name": "","total_time": 0,"call_count": 0,"is_running": 0},"children":[0]},
    //              {"key":0,"name":"void modmesh::detail::foo1()","data":{"start_time": 17745276708555250,"caller_name": "void modmesh::detail::foo1()","total_time": 61002916,"call_count": 1,"is_running": 1},"children":[1]},
    //              {"key":1,"name":"void modmesh::detail::foo2()","data":{"start_time": 17745276708555458,"caller_name": "void modmesh::detail::foo2()","total_time": 54002250,"call_count": 1,"is_running": 1},"children":[2]},
    //              {"key":2,"name":"void modmesh::detail::foo3()","data":{"start_time": 17745276743555833,"caller_name": "void modmesh::detail::foo3()","total_time": 19001833,"call_count": 1,"is_running": 1},"children":[]}
    //          ],
    //          "current_node":-1,
    //          "unique_id":3
    //      }
    //  }

    std::stringstream ss;
    CallProfilerSerializer::serialize(*pProfiler, ss);

    int ss_start_time_pos0 = ss.str().find(start_time_str);
    int ss_start_time_pos1 = ss.str().find(start_time_str, ss_start_time_pos0 + 1);
    int ss_start_time_pos2 = ss.str().find(start_time_str, ss_start_time_pos1 + 1);
    int ss_start_time_pos3 = ss.str().find(start_time_str, ss_start_time_pos2 + 1);
    int ss_total_time_pos0 = ss.str().find(total_time_str);
    int ss_total_time_pos1 = ss.str().find(total_time_str, ss_total_time_pos0 + 1);
    int ss_total_time_pos2 = ss.str().find(total_time_str, ss_total_time_pos1 + 1);
    int ss_total_time_pos3 = ss.str().find(total_time_str, ss_total_time_pos2 + 1);
    int ss_caller_name_pos0 = ss.str().find(caller_name_str);
    int ss_caller_name_pos1 = ss.str().find(caller_name_str, ss_caller_name_pos0 + 1);
    int ss_caller_name_pos2 = ss.str().find(caller_name_str, ss_caller_name_pos1 + 1);
    int ss_caller_name_pos3 = ss.str().find(caller_name_str, ss_caller_name_pos2 + 1);
    int ss_call_count_pos0 = ss.str().find(call_count_str);
    int ss_call_count_pos1 = ss.str().find(call_count_str, ss_call_count_pos0 + 1);
    int ss_call_count_pos2 = ss.str().find(call_count_str, ss_call_count_pos1 + 1);
    int ss_call_count_pos3 = ss.str().find(call_count_str, ss_call_count_pos2 + 1);

    int example_start_time_pos0 = serializeExample.find(start_time_str);
    int example_start_time_pos1 = serializeExample.find(start_time_str, example_start_time_pos0 + 1);
    int example_start_time_pos2 = serializeExample.find(start_time_str, example_start_time_pos1 + 1);
    int example_start_time_pos3 = serializeExample.find(start_time_str, example_start_time_pos2 + 1);
    int example_total_time_pos0 = serializeExample.find(total_time_str);
    int example_total_time_pos1 = serializeExample.find(total_time_str, example_total_time_pos0 + 1);
    int example_total_time_pos2 = serializeExample.find(total_time_str, example_total_time_pos1 + 1);
    int example_total_time_pos3 = serializeExample.find(total_time_str, example_total_time_pos2 + 1);
    int example_caller_name_pos0 = serializeExample.find(caller_name_str);
    int example_caller_name_pos1 = serializeExample.find(caller_name_str, example_caller_name_pos0 + 1);
    int example_caller_name_pos2 = serializeExample.find(caller_name_str, example_caller_name_pos1 + 1);
    int example_caller_name_pos3 = serializeExample.find(caller_name_str, example_caller_name_pos2 + 1);
    int example_call_count_pos0 = serializeExample.find(call_count_str);
    int example_call_count_pos1 = serializeExample.find(call_count_str, example_call_count_pos0 + 1);
    int example_call_count_pos2 = serializeExample.find(call_count_str, example_call_count_pos1 + 1);
    int example_call_count_pos3 = serializeExample.find(call_count_str, example_call_count_pos2 + 1);

    // Validate the serialization result except for the start_time and total_time
    EXPECT_EQ(ss.str().substr(0, ss_start_time_pos0 + start_time_str.size()), serializeExample.substr(0, example_start_time_pos0 + start_time_str.size()));
    EXPECT_EQ(ss.str().substr(ss_caller_name_pos0, ss_total_time_pos0 + total_time_str.size() - ss_caller_name_pos0), serializeExample.substr(example_caller_name_pos0, example_total_time_pos0 + total_time_str.size() - example_caller_name_pos0));
    EXPECT_EQ(ss.str().substr(ss_call_count_pos0, ss_start_time_pos1 - ss_call_count_pos0), serializeExample.substr(example_call_count_pos0, example_start_time_pos1 - example_call_count_pos0));
    EXPECT_EQ(ss.str().substr(ss_caller_name_pos1, ss_total_time_pos1 + total_time_str.size() - ss_caller_name_pos1), serializeExample.substr(example_caller_name_pos1, example_total_time_pos1 + total_time_str.size() - example_caller_name_pos1));
    EXPECT_EQ(ss.str().substr(ss_call_count_pos1, ss_start_time_pos2 - ss_call_count_pos1), serializeExample.substr(example_call_count_pos1, example_start_time_pos2 - example_call_count_pos1));
    EXPECT_EQ(ss.str().substr(ss_caller_name_pos2, ss_total_time_pos2 + total_time_str.size() - ss_caller_name_pos2), serializeExample.substr(example_caller_name_pos2, example_total_time_pos2 + total_time_str.size() - example_caller_name_pos2));
    EXPECT_EQ(ss.str().substr(ss_call_count_pos2, ss_start_time_pos3 - ss_call_count_pos2), serializeExample.substr(example_call_count_pos2, example_start_time_pos3 - example_call_count_pos2));
    EXPECT_EQ(ss.str().substr(ss_caller_name_pos3, ss_total_time_pos3 + total_time_str.size() - ss_caller_name_pos3), serializeExample.substr(example_caller_name_pos3, example_total_time_pos3 + total_time_str.size() - example_caller_name_pos3));
    EXPECT_EQ(ss.str().substr(ss_call_count_pos3), serializeExample.substr(example_call_count_pos3));
}

} // namespace detail
} // namespace modmesh