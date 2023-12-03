#include <gtest/gtest.h>
#include <thread>

#include <modmesh/toggle/profileRadixTree.hpp>

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
        TimeRegistryRadixTree & reg = TimeRegistryRadixTree::me();
        info = &reg;
    }

    TimeRegistryRadixTree * info;
};

constexpr int uniqueTime1 = 19;
constexpr int uniqueTime2 = 35;
constexpr int uniqueTime3 = 7;

class Foo
{
public:
    Foo() = default;

    void bar3()
    {
        MODMESH_TIME_RADIX();
        std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime1));
    }

    void bar2()
    {
        MODMESH_TIME_RADIX();
        std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime2));
        bar3();
    }

    void bar1()
    {
        MODMESH_TIME_RADIX();
        bar2();
        std::this_thread::sleep_for(std::chrono::milliseconds(uniqueTime3));
    }
};

TEST_F(CallProfilerTest, simple_case_2)
{
    info->reset();
    modmesh::Foo f;
    f.bar1();
    f.bar2();
    f.bar3();
    f.bar3();
    info->report();

}

} // namespace modmesh