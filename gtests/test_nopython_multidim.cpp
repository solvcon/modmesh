#include <solvcon/solvcon.hpp>
#include <solvcon/multidim/multidim.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

using namespace solvcon;

namespace
{

std::shared_ptr<StaticMesh> build_mesh(
    uint8_t ndim,
    std::vector<std::array<double, 3>> const & coords,
    std::vector<int32_t> const & cltpn,
    std::vector<std::vector<int32_t>> const & clnds)
{
    auto const nnode = static_cast<StaticMesh::uint_type>(coords.size());
    auto const ncell = static_cast<StaticMesh::uint_type>(cltpn.size());
    auto mh = StaticMesh::construct(ndim, nnode, StaticMesh::uint_type(0), ncell);
    for (StaticMesh::uint_type ind = 0; ind < nnode; ++ind)
    {
        for (uint8_t d = 0; d < ndim; ++d)
        {
            mh->ndcrd(ind, d) = coords[ind][d];
        }
    }
    for (StaticMesh::uint_type icl = 0; icl < ncell; ++icl)
    {
        mh->cltpn(icl) = cltpn[icl];
        for (size_t j = 0; j < clnds[icl].size(); ++j)
        {
            mh->clnds(static_cast<int32_t>(icl), static_cast<int32_t>(j)) = clnds[icl][j];
        }
    }
    mh->build_interior(true);
    mh->build_boundary();
    mh->build_ghost();
    return mh;
}

// Unit-square quadrilateral.
std::shared_ptr<StaticMesh> make_quad()
{
    return build_mesh(
        2,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}},
        {CellType::QUADRILATERAL},
        {{4, 0, 1, 2, 3}});
}

// Three triangles around the origin.
std::shared_ptr<StaticMesh> make_triangles()
{
    return build_mesh(
        2,
        {{0, 0, 0}, {-1, -1, 0}, {1, -1, 0}, {0, 1, 0}},
        {CellType::TRIANGLE, CellType::TRIANGLE, CellType::TRIANGLE},
        {{3, 0, 1, 2}, {3, 0, 2, 3}, {3, 0, 3, 1}});
}

// One quadrilateral and two triangles.
std::shared_ptr<StaticMesh> make_mixed()
{
    return build_mesh(
        2,
        {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}},
        {CellType::QUADRILATERAL, CellType::TRIANGLE, CellType::TRIANGLE},
        {{4, 0, 1, 4, 3}, {3, 1, 2, 4, 0}, {3, 2, 5, 4, 0}});
}

} /* end namespace */

TEST(Multidim, ge_linalg_2d)
{
    GradientElement::ge_matrix_type a = {};
    a[0] = {2, 1, 0};
    a[1] = {1, 3, 0};

    EXPECT_DOUBLE_EQ(5.0, GradientElement::determinant(a, 2));

    GradientElement::ge_matrix_type const adj = GradientElement::adjugate(a, 2);
    EXPECT_DOUBLE_EQ(3.0, adj[0][0]);
    EXPECT_DOUBLE_EQ(-1.0, adj[0][1]);
    EXPECT_DOUBLE_EQ(-1.0, adj[1][0]);
    EXPECT_DOUBLE_EQ(2.0, adj[1][1]);

    // GradientElement::adjugate(a) * a == GradientElement::determinant(a) * I.
    for (size_t i = 0; i < 2; ++i)
    {
        GradientElement::ge_vector_type const col = {a[0][i], a[1][i], 0};
        GradientElement::ge_vector_type const r = GradientElement::multiply(adj, col, 2);
        EXPECT_NEAR((0 == i) ? 5.0 : 0.0, r[0], 1e-12);
        EXPECT_NEAR((1 == i) ? 5.0 : 0.0, r[1], 1e-12);
    }
}

TEST(Multidim, ge_linalg_3d)
{
    GradientElement::ge_matrix_type a = {};
    a[0] = {1, 2, 3};
    a[1] = {0, 1, 4};
    a[2] = {5, 6, 0};

    EXPECT_DOUBLE_EQ(1.0, GradientElement::determinant(a, 3));

    GradientElement::ge_matrix_type const adj = GradientElement::adjugate(a, 3);
    GradientElement::ge_matrix_type const expect = {{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}}};
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(expect[i][j], adj[i][j]);
        }
    }

    // Solve a x = b via x = GradientElement::adjugate(a) b / det(a) and verify a x == b.
    GradientElement::ge_vector_type const b = {6, -1, 2};
    GradientElement::ge_vector_type const adjb = GradientElement::multiply(adj, b, 3);
    double const det = GradientElement::determinant(a, 3);
    GradientElement::ge_vector_type const x = {adjb[0] / det, adjb[1] / det, adjb[2] / det};
    GradientElement::ge_vector_type const ax = GradientElement::multiply(a, x, 3);
    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(b[i], ax[i], 1e-12);
    }
}

TEST(Multidim, displacement_matrix_nonsingular)
{
    std::vector<std::shared_ptr<StaticMesh>> const meshes = {
        make_quad(), make_triangles(), make_mixed()};
    for (auto const & mh : meshes)
    {
        auto const ec = EulerCore::construct(mh, 0.01);
        SimpleArray<double> const & cecnd = ec->cecnd();
        for (int32_t icl = 0; icl < static_cast<int32_t>(mh->ncell()); ++icl)
        {
            GradientElement const ge(*mh, cecnd, icl, 1.0);
            for (int32_t ifge = 0; ifge < ge.nfge(); ++ifge)
            {
                GradientElement::ge_matrix_type const dst = ge.displacement_matrix(ifge);
                EXPECT_GT(std::fabs(GradientElement::determinant(dst, mh->ndim())), 1e-10)
                    << "singular FGE matrix: cell " << icl << " ifge " << ifge;
            }
        }
    }
}

TEST(Multidim, solve_gradient_linear_field)
{
    // For a linear field u(x) = c + g . x, the solution delta at each gradient
    // evaluation point is exactly g . idis, so the reconstructed gradient must
    // recover g exactly (up to round-off).
    GradientElement::ge_vector_type const g = {1.5, -2.7, 0.0};
    std::vector<std::shared_ptr<StaticMesh>> const meshes = {
        make_quad(), make_triangles(), make_mixed()};
    for (auto const & mh : meshes)
    {
        size_t const ndim = mh->ndim();
        auto const ec = EulerCore::construct(mh, 0.01);
        SimpleArray<double> const & cecnd = ec->cecnd();
        for (int32_t icl = 0; icl < static_cast<int32_t>(mh->ncell()); ++icl)
        {
            GradientElement const ge(*mh, cecnd, icl, 1.0);
            for (int32_t ifge = 0; ifge < ge.nfge(); ++ifge)
            {
                GradientElement::ge_matrix_type const dst = ge.displacement_matrix(ifge);
                auto const & tface = ge.faces(ifge);
                // Displacement-matrix rows must equal the per-face idis vectors.
                for (size_t ivx = 0; ivx < ndim; ++ivx)
                {
                    for (size_t d = 0; d < ndim; ++d)
                    {
                        EXPECT_DOUBLE_EQ(ge.idis(tface[ivx] - 1, static_cast<int32_t>(d)), dst[ivx][d]);
                    }
                }
                GradientElement::ge_vector_type const udf = GradientElement::multiply(dst, g, ndim);
                GradientElement::ge_vector_type const got = ge.solve_gradient(ifge, udf);
                for (size_t d = 0; d < ndim; ++d)
                {
                    EXPECT_NEAR(g[d], got[d], 1e-9)
                        << "cell " << icl << " ifge " << ifge << " dim " << d;
                }
            }
        }
    }
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
