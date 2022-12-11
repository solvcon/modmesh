#include <modmesh/modmesh.hpp>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(nopython_modmesh, dummy)
{
    EXPECT_TRUE(true);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
