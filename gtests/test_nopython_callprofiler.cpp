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
    /* Here is the expected format of output:
    {
        "radix_tree": {
            "current_node": "105553178001632",
            "nodes": [{
                    "key": -1,
                    "name": "",
                    "data": {
                        "start_time": 0,
                        "caller_name": "",
                        "total_time": 0,
                        "call_count": 0,
                        "is_running": 0
                    },
                    "address": "105553178001632",
                    children": [
                        "105553178002080"
                    ]
                },
                {
                    "key": 0,
                    "name": "void modmesh::detail::foo1()",
                    "data": {
                        "start_time": 1969175311810791,
                        "caller_name": "void modmesh::detail::foo1()",
                        "total_time": 61010334,
                        "call_count": 1,
                        "is_running": 1
                    },
                    "address": "105553178002080",
                    children": [
                        "105553178001408"
                    ]
                },
                {
                    "key": 1,
                    "name": "void modmesh::detail::foo2()",
                    "data": {
                        "start_time": 1969175311811000,
                        "caller_name": "void modmesh::detail::foo2()",
                        "total_time": 54010041,
                        "call_count": 1,
                        "is_running": 1
                    },
                    "address": "105553178001408",
                    children": [
                        "105553178034176"
                    ]
                },
                {
                    "key": 2,
                    "name": "void modmesh::detail::foo3()",
                    "data": {
                        "start_time": 1969175346815166,
                        "caller_name": "void modmesh::detail::foo3()",
                        "total_time": 19005625,
                        "call_count": 1,
                        "is_running": 1
                    },
                    "address": "105553178034176",
                    children": []
                }
            ],
            "id_map": {
                "void modmesh::detail::foo3()": 2,
                "void modmesh::detail::foo2()": 1,
                "void modmesh::detail::foo1()": 0
            },
            "unique_id": 3
        }
    }
    */
    pProfiler->reset();

    foo1();
    std::stringstream ss;
    CallProfilerSerializer::serialize(*pProfiler, ss);
    std::cout << ss.str() << std::endl;
}

} // namespace detail
} // namespace modmesh