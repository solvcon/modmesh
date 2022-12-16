#include <modmesh/buffer/buffer.hpp>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(nopython_buffer, dummy)
{
    EXPECT_TRUE(true);
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

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
