/*
 * Copyright (c) 2025, An-Chi Liu <phy.tiger@gmail.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>
#include <modmesh/universe/rtree.hpp>

#include <random>
#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

// Define a simple value operations for 2D points
struct Point2D
{
    double x;
    double y;

    bool operator==(Point2D const & other) const
    {
        return x == other.x && y == other.y;
    }
};

using TestBoundBox2d = modmesh::BoundBox2d<double>;

struct Point2DValueOps
{
    static TestBoundBox2d calc_bound_box(Point2D const & item)
    {
        return TestBoundBox2d(item.x, item.y, item.x, item.y);
    }

    static TestBoundBox2d calc_group_bound_box(std::vector<Point2D> const & items)
    {
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double max_y = std::numeric_limits<double>::lowest();

        for (const auto & item : items)
        {
            min_x = std::min(min_x, item.x);
            min_y = std::min(min_y, item.y);
            max_x = std::max(max_x, item.x);
            max_y = std::max(max_y, item.y);
        }
        return TestBoundBox2d(min_x, min_y, max_x, max_y);
    }
};

struct Point3D
{
    double x;
    double y;
    double z;

    bool operator==(Point3D const & other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

using TestBoundBox3d = modmesh::BoundBox3d<double>;
struct Point3DValueOps
{
    static TestBoundBox3d calc_bound_box(Point3D const & item)
    {
        return TestBoundBox3d(item.x, item.y, item.z, item.x, item.y, item.z);
    }

    static TestBoundBox3d calc_group_bound_box(std::vector<Point3D> const & items)
    {
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double min_z = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double max_y = std::numeric_limits<double>::lowest();
        double max_z = std::numeric_limits<double>::lowest();

        for (const auto & item : items)
        {
            min_x = std::min(min_x, item.x);
            min_y = std::min(min_y, item.y);
            min_z = std::min(min_z, item.z);
            max_x = std::max(max_x, item.x);
            max_y = std::max(max_y, item.y);
            max_z = std::max(max_z, item.z);
        }
        return TestBoundBox3d(min_x, min_y, min_z, max_x, max_y, max_z);
    }
};

TEST(RTree, basic_operations)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps> rtree;

    rtree.insert(Point2D{1.0, 1.0});
    rtree.insert(Point2D{2.0, 2.0});
    rtree.insert(Point2D{3.0, 3.0});

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(1.5, 1.5, 2.5, 2.5), results);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].x, 2.0);
    EXPECT_EQ(results[0].y, 2.0);

    rtree.remove(Point2D{2.0, 2.0});
}

TEST(RTree, basic_operations_3d)
{
    using namespace modmesh;

    RTree<Point3D, TestBoundBox3d, Point3DValueOps> rtree;

    rtree.insert(Point3D{1.0, 1.0, 1.0});
    rtree.insert(Point3D{2.0, 2.0, 2.0});
    rtree.insert(Point3D{3.0, 3.0, 3.0});

    std::vector<Point3D> results;
    rtree.search(TestBoundBox3d(1.5, 1.5, 1.5, 2.5, 2.5, 2.5), results);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].x, 2.0);
    EXPECT_EQ(results[0].y, 2.0);
    EXPECT_EQ(results[0].z, 2.0);

    rtree.remove(Point3D{2.0, 2.0, 2.0});
}

TEST(RTree, complex_operations)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps, 4> rtree; // Small max items per node to force splits

    // Insert multiple points
    for (int i = 0; i < 100; ++i)
    {
        rtree.insert(Point2D{static_cast<double>(i), static_cast<double>(i)});
    }

    // Search for a range
    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(20.0, 20.0, 30.0, 30.0), results);
    EXPECT_EQ(results.size(), 11); // Points from (20,20) to (30,30)

    // Remove a point and verify it's gone
    rtree.remove(Point2D{25.0, 25.0});
    results.clear();
    rtree.search(TestBoundBox2d(20.0, 20.0, 30.0, 30.0), results);
    EXPECT_EQ(results.size(), 10); // One less after removal
}

TEST(RTree, empty_tree_operations)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps> rtree;

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(0.0, 0.0, 10.0, 10.0), results);
    EXPECT_EQ(results.size(), 0);

    rtree.remove(Point2D{1.0, 1.0});
    EXPECT_EQ(results.size(), 0);
}

TEST(RTree, single_point_operations)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps> rtree;

    rtree.insert(Point2D{5.0, 5.0});

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(0.0, 0.0, 10.0, 10.0), results);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].x, 5.0);
    EXPECT_EQ(results[0].y, 5.0);

    rtree.remove(Point2D{5.0, 5.0});
    results.clear();
    rtree.search(TestBoundBox2d(0.0, 0.0, 10.0, 10.0), results);
    EXPECT_EQ(results.size(), 0);
}

TEST(RTree, overlapping_search_regions)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps> rtree;

    rtree.insert(Point2D{5.0, 5.0});
    rtree.insert(Point2D{15.0, 15.0});
    rtree.insert(Point2D{25.0, 25.0});

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(0.0, 0.0, 20.0, 20.0), results);
    EXPECT_EQ(results.size(), 2);

    results.clear();
    rtree.search(TestBoundBox2d(10.0, 10.0, 30.0, 30.0), results);
    EXPECT_EQ(results.size(), 2);
}

TEST(RTree, boundary_conditions)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps> rtree;

    rtree.insert(Point2D{10.0, 10.0});
    rtree.insert(Point2D{20.0, 20.0});

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(10.0, 10.0, 20.0, 20.0), results);
    EXPECT_EQ(results.size(), 2);

    results.clear();
    rtree.search(TestBoundBox2d(10.1, 10.1, 19.9, 19.9), results);
    EXPECT_EQ(results.size(), 0);
}

TEST(RTree, large_scale_insertion_2d)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps, 8> rtree;

    for (int i = 0; i < 1000; ++i)
    {
        rtree.insert(Point2D{static_cast<double>(i % 100), static_cast<double>(i / 100)});
    }

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(0.0, 0.0, 10.0, 10.0), results);
    EXPECT_GT(results.size(), 0);
}

TEST(RTree, random_insertion_and_search_2d)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps, 16> rtree;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::vector<Point2D> inserted_points;
    for (int i = 0; i < 200; ++i)
    {
        Point2D pt{dis(gen), dis(gen)};
        inserted_points.push_back(pt);
        rtree.insert(pt);
    }

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(40.0, 40.0, 60.0, 60.0), results);

    int expected_count = 0;
    for (const auto & pt : inserted_points)
    {
        if (pt.x >= 40.0 && pt.x <= 60.0 && pt.y >= 40.0 && pt.y <= 60.0)
        {
            ++expected_count;
        }
    }
    EXPECT_EQ(results.size(), expected_count);
}

TEST(RTree, multiple_removals_2d)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps, 4> rtree;

    for (int i = 0; i < 50; ++i)
    {
        rtree.insert(Point2D{static_cast<double>(i), static_cast<double>(i)});
    }

    for (int i = 0; i < 25; ++i)
    {
        rtree.remove(Point2D{static_cast<double>(i * 2), static_cast<double>(i * 2)});
    }

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(0.0, 0.0, 50.0, 50.0), results);
    EXPECT_EQ(results.size(), 25);
}

TEST(RTree, scattered_points_2d)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps> rtree;

    rtree.insert(Point2D{0.0, 0.0});
    rtree.insert(Point2D{100.0, 100.0});
    rtree.insert(Point2D{-50.0, 50.0});
    rtree.insert(Point2D{50.0, -50.0});

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(-60.0, -60.0, 110.0, 110.0), results);
    EXPECT_EQ(results.size(), 4);

    results.clear();
    rtree.search(TestBoundBox2d(-10.0, -10.0, 10.0, 10.0), results);
    EXPECT_EQ(results.size(), 1);
}

TEST(RTree, duplicate_points_2d)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps> rtree;

    rtree.insert(Point2D{5.0, 5.0});
    rtree.insert(Point2D{5.0, 5.0});
    rtree.insert(Point2D{5.0, 5.0});

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(0.0, 0.0, 10.0, 10.0), results);
    EXPECT_EQ(results.size(), 3);

    rtree.remove(Point2D{5.0, 5.0});
    results.clear();
    rtree.search(TestBoundBox2d(0.0, 0.0, 10.0, 10.0), results);
    EXPECT_EQ(results.size(), 2);
}

TEST(RTree, large_scale_insertion_3d)
{
    using namespace modmesh;

    RTree<Point3D, TestBoundBox3d, Point3DValueOps, 8> rtree;

    for (int i = 0; i < 500; ++i)
    {
        rtree.insert(Point3D{static_cast<double>(i % 10),
                             static_cast<double>((i / 10) % 10),
                             static_cast<double>(i / 100)});
    }

    std::vector<Point3D> results;
    rtree.search(TestBoundBox3d(0.0, 0.0, 0.0, 5.0, 5.0, 5.0), results);
    EXPECT_GT(results.size(), 0);
}

TEST(RTree, random_insertion_and_search_3d)
{
    using namespace modmesh;

    RTree<Point3D, TestBoundBox3d, Point3DValueOps, 16> rtree;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::vector<Point3D> inserted_points;
    for (int i = 0; i < 150; ++i)
    {
        Point3D pt{dis(gen), dis(gen), dis(gen)};
        inserted_points.push_back(pt);
        rtree.insert(pt);
    }

    std::vector<Point3D> results;
    rtree.search(TestBoundBox3d(30.0, 30.0, 30.0, 70.0, 70.0, 70.0), results);

    int expected_count = 0;
    for (const auto & pt : inserted_points)
    {
        if (pt.x >= 30.0 && pt.x <= 70.0 &&
            pt.y >= 30.0 && pt.y <= 70.0 &&
            pt.z >= 30.0 && pt.z <= 70.0)
        {
            ++expected_count;
        }
    }
    EXPECT_EQ(results.size(), expected_count);
}

TEST(RTree, corner_cases_3d)
{
    using namespace modmesh;

    RTree<Point3D, TestBoundBox3d, Point3DValueOps> rtree;

    rtree.insert(Point3D{0.0, 0.0, 0.0});
    rtree.insert(Point3D{100.0, 100.0, 100.0});
    rtree.insert(Point3D{-50.0, 50.0, 0.0});

    std::vector<Point3D> results;
    rtree.search(TestBoundBox3d(-60.0, -10.0, -10.0, 110.0, 110.0, 110.0), results);
    EXPECT_EQ(results.size(), 3);

    results.clear();
    rtree.search(TestBoundBox3d(-5.0, -5.0, -5.0, 5.0, 5.0, 5.0), results);
    EXPECT_EQ(results.size(), 1);
}

TEST(RTree, sequential_removal_3d)
{
    using namespace modmesh;

    RTree<Point3D, TestBoundBox3d, Point3DValueOps, 4> rtree;

    for (int i = 0; i < 30; ++i)
    {
        rtree.insert(Point3D{static_cast<double>(i),
                             static_cast<double>(i),
                             static_cast<double>(i)});
    }

    for (int i = 0; i < 15; ++i)
    {
        rtree.remove(Point3D{static_cast<double>(i * 2),
                             static_cast<double>(i * 2),
                             static_cast<double>(i * 2)});
    }

    std::vector<Point3D> results;
    rtree.search(TestBoundBox3d(0.0, 0.0, 0.0, 30.0, 30.0, 30.0), results);
    EXPECT_EQ(results.size(), 15);
}

TEST(RTree, stress_test_insert_remove_2d)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps, 8> rtree;
    std::mt19937 gen(999);
    std::uniform_real_distribution<> dis(0.0, 1000.0);

    std::vector<Point2D> points;
    for (int i = 0; i < 300; ++i)
    {
        Point2D pt{dis(gen), dis(gen)};
        points.push_back(pt);
        rtree.insert(pt);
    }

    for (size_t i = 0; i < points.size() / 2; ++i)
    {
        rtree.remove(points[i]);
    }

    std::vector<Point2D> results;
    rtree.search(TestBoundBox2d(0.0, 0.0, 1000.0, 1000.0), results);
    EXPECT_EQ(results.size(), points.size() / 2);
}

TEST(RTree, non_overlapping_searches_2d)
{
    using namespace modmesh;

    RTree<Point2D, TestBoundBox2d, Point2DValueOps> rtree;

    for (int i = 0; i < 10; ++i)
    {
        rtree.insert(Point2D{static_cast<double>(i * 10), static_cast<double>(i * 10)});
    }

    std::vector<Point2D> results1;
    rtree.search(TestBoundBox2d(0.0, 0.0, 25.0, 25.0), results1);

    std::vector<Point2D> results2;
    rtree.search(TestBoundBox2d(50.0, 50.0, 100.0, 100.0), results2);

    EXPECT_GT(results1.size(), 0);
    EXPECT_GT(results2.size(), 0);

    for (const auto & r1 : results1)
    {
        for (const auto & r2 : results2)
        {
            EXPECT_FALSE(r1 == r2);
        }
    }
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
