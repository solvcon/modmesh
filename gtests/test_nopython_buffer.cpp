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

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
