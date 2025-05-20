#include <modmesh/buffer/buffer.hpp>

#include <gtest/gtest.h>

#include <random>
#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(ConcreteBuffer, iterator)
{
    using namespace modmesh;

    auto buffer = ConcreteBuffer::construct(10);
    int8_t i = 0;
    for (auto & it : *buffer)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : *buffer)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(SimpleArray, construction)
{
    namespace mm = modmesh;
    mm::SimpleArray<double> arr_double(10);
    EXPECT_EQ(arr_double.nbody(), 10);
    mm::SimpleArray<int> arr_int(17);
    EXPECT_EQ(arr_int.nbody(), 17);
}

TEST(SimpleArray, minmaxsum)
{
    using namespace modmesh;

    SimpleArray<double> arr_double(small_vector<size_t>{10}, 0);
    EXPECT_EQ(arr_double.sum(), 0);
    EXPECT_EQ(arr_double.min(), 0);
    EXPECT_EQ(arr_double.max(), 0);
    arr_double.fill(3.14);
    EXPECT_EQ(arr_double.sum(), 3.14 * 10);
    EXPECT_EQ(arr_double.min(), 3.14);
    EXPECT_EQ(arr_double.max(), 3.14);
    arr_double(2) = -2.9;
    arr_double(4) = 12.7;
    EXPECT_EQ(arr_double.min(), -2.9);
    EXPECT_EQ(arr_double.max(), 12.7);

    SimpleArray<int> arr_int(small_vector<size_t>{3, 4}, -2);
    EXPECT_EQ(arr_int.sum(), -2 * 3 * 4);
    EXPECT_EQ(arr_int.min(), -2);
    EXPECT_EQ(arr_int.max(), -2);
    arr_int.fill(7);
    EXPECT_EQ(arr_int.sum(), 7 * 3 * 4);
    EXPECT_EQ(arr_int.min(), 7);
    EXPECT_EQ(arr_int.max(), 7);
    arr_int(1, 2) = -8;
    arr_int(2, 0) = 9;
    EXPECT_EQ(arr_int.min(), -8);
    EXPECT_EQ(arr_int.max(), 9);
}

TEST(SimpleArray, abs)
{
    using namespace modmesh;

    SimpleArray<double> arr(small_vector<size_t>{10}, -1.0);
    EXPECT_EQ(arr.sum(), -10.0);

    SimpleArray<double> brr = arr.abs();
    EXPECT_EQ(brr.sum(), 10.0);
}

TEST(SimpleArray, iterator)
{
    using namespace modmesh;

    SimpleArray<double> arr(10);
    int8_t i = 0;
    for (auto & it : arr)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : arr)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(SimpleArray_DataType, from_type)
{
    modmesh::DataType dt_double = modmesh::DataType::from<double>();
    EXPECT_EQ(dt_double.type(), modmesh::DataType::Float64);

    modmesh::DataType dt_int = modmesh::DataType::from<int>();
    EXPECT_EQ(dt_int.type(), modmesh::DataType::Int32);
}

TEST(SimpleArray_DataType, from_string)
{
    modmesh::DataType dt_double = modmesh::DataType("float64");
    EXPECT_EQ(dt_double.type(), modmesh::DataType::Float64);

    modmesh::DataType dt_bool = modmesh::DataType("bool");
    EXPECT_EQ(dt_bool.type(), modmesh::DataType::Bool);

    EXPECT_THROW(modmesh::DataType("float16"), std::invalid_argument); // float16 does not exist
    EXPECT_THROW(modmesh::DataType("bool8"), std::invalid_argument); // bool8 does not exist
}

TEST(BufferExpander, iterator)
{
    using namespace modmesh;

    auto buffer = BufferExpander::construct(10);
    int8_t i = 0;
    for (auto & it : *buffer)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : *buffer)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(small_vector, select_kth)
{
    const size_t n = 1024;
    std::vector<int> scrambled(n);
    // gcd(31, 1024) = 1,
    // so we can get all numbers from 0 to 1023.
    for (size_t i = 0; i < n; ++i)
    {
        scrambled[i] = static_cast<int>((i * 31) % n);
    }

    for (size_t k = 0; k < n; ++k)
    {
        modmesh::small_vector<int> sv(scrambled);
        int result = sv.select_kth(k);
        EXPECT_EQ(result, static_cast<int>(k));
    }
}

TEST(small_vector, select_kth_random)
{
    size_t n = 1024;
    std::vector<int> vec(n);
    std::iota(vec.begin(), vec.end(), 0);

    modmesh::small_vector<int> sv(vec);
    for (size_t i = 0; i < n; ++i)
    {
        auto rng = std::default_random_engine{};
        std::shuffle(sv.begin(), sv.end(), rng);
        int it = sv.select_kth(i);
        EXPECT_EQ(it, i);
    }
}
// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
