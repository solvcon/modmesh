/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
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

/**
 * Boundary conditions for the Euler CESE solver: the ghost-cell trim passes
 * bc_soln (order 0) and bc_dsoln (order 1). Each handler writes the ghost row
 * of every boundary face it owns, rotating into the face-normal frame given by
 * get_normal_matrix.  NonReflective also realizes (supersonic) outflow;
 * SlipWall mirrors momentum and its derivatives across the wall; Inlet fixes a
 * free-stream conserved state.
 */

#include <modmesh/multidim/euler.hpp>

#include <array>
#include <cmath>
#include <format>
#include <stdexcept>

namespace modmesh
{

namespace detail
{

// Fixed-size dense vector/matrix on the leading ndim block (ndim is 2 or 3);
// the boundary rotations are tiny, so plain std::array beats the dynamic
// modmesh::small_vector / BLAS wrappers, as in GradientElement.
template <size_t NDIM>
using bc_vector_type = std::array<double, NDIM>;
template <size_t NDIM>
using bc_matrix_type = std::array<bc_vector_type<NDIM>, NDIM>;

// ret[i] = sum_j m[i][j] * v[j]
template <size_t NDIM>
bc_vector_type<NDIM> bc_matvec(bc_matrix_type<NDIM> const & m, bc_vector_type<NDIM> const & v)
{
    bc_vector_type<NDIM> ret = {};
    for (size_t i = 0; i < NDIM; ++i)
    {
        for (size_t j = 0; j < NDIM; ++j)
        {
            ret[i] += m[i][j] * v[j];
        }
    }
    return ret;
}

template <size_t NDIM>
bc_matrix_type<NDIM> bc_transpose(bc_matrix_type<NDIM> const & m)
{
    bc_matrix_type<NDIM> ret = {};
    for (size_t i = 0; i < NDIM; ++i)
    {
        for (size_t j = 0; j < NDIM; ++j)
        {
            ret[i][j] = m[j][i];
        }
    }
    return ret;
}

// ret[i][j] = sum_k a[i][k] * b[k][j]
template <size_t NDIM>
bc_matrix_type<NDIM> bc_matmat(bc_matrix_type<NDIM> const & a, bc_matrix_type<NDIM> const & b)
{
    bc_matrix_type<NDIM> ret = {};
    for (size_t i = 0; i < NDIM; ++i)
    {
        for (size_t j = 0; j < NDIM; ++j)
        {
            for (size_t k = 0; k < NDIM; ++k)
            {
                ret[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return ret;
}

// Orthonormal frame whose first row is the outward unit normal n.  The choice
// of tangent rows does not affect any handler: each handler rotates by the
// matrix and back by its transpose, so the tangent basis cancels and only the
// wall-normal reflection (independent of the basis) survives.
template <size_t NDIM>
bc_matrix_type<NDIM> normal_matrix(bc_vector_type<NDIM> const & n);

template <>
bc_matrix_type<2> normal_matrix<2>(bc_vector_type<2> const & n)
{
    return {{{n[0], n[1]}, {-n[1], n[0]}}};
}

template <>
bc_matrix_type<3> normal_matrix<3>(bc_vector_type<3> const & n)
{
    // Build the tangent from the coordinate axis least aligned with the
    // normal, so the cross product stays well conditioned for any normal.
    bc_vector_type<3> axis = {0.0, 0.0, 0.0};
    double const ax = std::fabs(n[0]);
    double const ay = std::fabs(n[1]);
    double const az = std::fabs(n[2]);
    if (ax <= ay && ax <= az)
    {
        axis = {1.0, 0.0, 0.0};
    }
    else if (ay <= az)
    {
        axis = {0.0, 1.0, 0.0};
    }
    else
    {
        axis = {0.0, 0.0, 1.0};
    }
    bc_vector_type<3> t1 = {
        n[1] * axis[2] - n[2] * axis[1],
        n[2] * axis[0] - n[0] * axis[2],
        n[0] * axis[1] - n[1] * axis[0]};
    double const len = std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    t1 = {t1[0] / len, t1[1] / len, t1[2] / len};
    bc_vector_type<3> const t2 = {
        n[1] * t1[2] - n[2] * t1[1],
        n[2] * t1[0] - n[0] * t1[2],
        n[0] * t1[1] - n[1] * t1[0]};
    return {{{n[0], n[1], n[2]}, {t1[0], t1[1], t1[2]}, {t2[0], t2[1], t2[2]}}};
}

template <size_t NDIM>
bc_vector_type<NDIM> face_normal(StaticMesh const & msh, int32_t ifc)
{
    bc_vector_type<NDIM> n = {};
    for (size_t d = 0; d < NDIM; ++d)
    {
        n[d] = msh.fcnml(ifc, d);
    }
    return n;
}

// Order-0 ghost update (trim_do0).  Writes the ghost row tfccls[1] = jcl of
// each boundary face from the interior row tfccls[0] = icl.
template <size_t NDIM>
// This helper needs to be refactored in the future, but now I need to keep it
// similar to the legacy code.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void bc_soln_impl(EulerCore & ec)
{
    constexpr size_t neq = NDIM + 2;
    auto const & msh = *ec.mesh();
    SimpleArray<double> & so0n = ec.so0n();

    for (EulerBoundary const & bnd : ec.boundaries())
    {
        for (size_t ibnd = 0; ibnd < bnd.faces.size(); ++ibnd)
        {
            int32_t const ifc = bnd.faces[ibnd];
            int32_t const icl = msh.fccls(ifc, 0);
            int32_t const jcl = msh.fccls(ifc, 1);
            switch (bnd.kind)
            {
            case EulerBC::NonReflective:
            {
                // Copy the whole interior state to the ghost.
                for (size_t ieq = 0; ieq < neq; ++ieq)
                {
                    so0n(jcl, ieq) = so0n(icl, ieq);
                }
                break;
            }
            case EulerBC::SlipWall:
            {
                bc_matrix_type<NDIM> const mat = normal_matrix<NDIM>(face_normal<NDIM>(msh, ifc));
                bc_matrix_type<NDIM> const matinv = bc_transpose<NDIM>(mat);
                // Copy density and energy; reflect the momentum by negating its
                // wall-normal component in the rotated frame.
                so0n(jcl, 0) = so0n(icl, 0);
                so0n(jcl, neq - 1) = so0n(icl, neq - 1);
                bc_vector_type<NDIM> mom = {};
                for (size_t d = 0; d < NDIM; ++d)
                {
                    mom[d] = so0n(icl, 1 + d);
                }
                bc_vector_type<NDIM> rot = bc_matvec<NDIM>(mat, mom);
                rot[0] = -rot[0];
                bc_vector_type<NDIM> const out = bc_matvec<NDIM>(matinv, rot);
                for (size_t d = 0; d < NDIM; ++d)
                {
                    so0n(jcl, 1 + d) = out[d];
                }
                break;
            }
            case EulerBC::Inlet:
            {
                // Fixed free-stream conserved state from [rho, v(ndim), p, ga].
                double const rho = bnd.value[0];
                double const p = bnd.value[1 + NDIM];
                double const ga = bnd.value[2 + NDIM];
                double vsq = 0.0;
                for (size_t d = 0; d < NDIM; ++d)
                {
                    vsq += bnd.value[1 + d] * bnd.value[1 + d];
                }
                so0n(jcl, 0) = rho;
                for (size_t d = 0; d < NDIM; ++d)
                {
                    so0n(jcl, 1 + d) = rho * bnd.value[1 + d];
                }
                so0n(jcl, neq - 1) = p / (ga - 1.0) + 0.5 * rho * vsq;
                break;
            }
            }
        }
    }
}

// Order-1 ghost update (trim_do1).  Reads the interior order-1 current
// solution so1c[icl] and writes the ghost order-1 new solution so1n[jcl].
template <size_t NDIM>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void bc_dsoln_impl(EulerCore & ec)
{
    constexpr size_t neq = NDIM + 2;
    auto const & msh = *ec.mesh();
    SimpleArray<double> & so1c = ec.so1c();
    SimpleArray<double> & so1n = ec.so1n();

    for (EulerBoundary const & bnd : ec.boundaries())
    {
        for (size_t ibnd = 0; ibnd < bnd.faces.size(); ++ibnd)
        {
            int32_t const ifc = bnd.faces[ibnd];
            int32_t const icl = msh.fccls(ifc, 0);
            int32_t const jcl = msh.fccls(ifc, 1);

            if (EulerBC::Inlet == bnd.kind)
            {
                // A fixed inlet state carries no gradient.
                for (size_t ieq = 0; ieq < neq; ++ieq)
                {
                    for (size_t d = 0; d < NDIM; ++d)
                    {
                        so1n(jcl, ieq, d) = 0.0;
                    }
                }
                continue;
            }

            bc_matrix_type<NDIM> const mat = normal_matrix<NDIM>(face_normal<NDIM>(msh, ifc));
            bc_matrix_type<NDIM> const matinv = bc_transpose<NDIM>(mat);

            if (EulerBC::NonReflective == bnd.kind)
            {
                // Zero the wall-normal derivative of every equation, keep the
                // tangential part.
                for (size_t ieq = 0; ieq < neq; ++ieq)
                {
                    bc_vector_type<NDIM> g = {};
                    for (size_t d = 0; d < NDIM; ++d)
                    {
                        g[d] = so1c(icl, ieq, d);
                    }
                    bc_vector_type<NDIM> rot = bc_matvec<NDIM>(mat, g);
                    rot[0] = 0.0;
                    bc_vector_type<NDIM> const out = bc_matvec<NDIM>(matinv, rot);
                    for (size_t d = 0; d < NDIM; ++d)
                    {
                        so1n(jcl, ieq, d) = out[d];
                    }
                }
                continue;
            }

            // SlipWall: reflect the wall-normal derivative of the scalar fields
            // (density col 0, energy col neq-1) and mirror the momentum-gradient
            // tensor across the wall.
            for (size_t const ieq : {static_cast<size_t>(0), neq - 1})
            {
                bc_vector_type<NDIM> g = {};
                for (size_t d = 0; d < NDIM; ++d)
                {
                    g[d] = so1c(icl, ieq, d);
                }
                bc_vector_type<NDIM> rot = bc_matvec<NDIM>(mat, g);
                rot[0] = -rot[0];
                bc_vector_type<NDIM> const out = bc_matvec<NDIM>(matinv, rot);
                for (size_t d = 0; d < NDIM; ++d)
                {
                    so1n(jcl, ieq, d) = out[d];
                }
            }
            // Momentum gradient mom_grad[a][b] = d(momentum_a)/d(x_b).  Rotate
            // both indices into the normal frame, negate the normal-tangential
            // cross terms, then rotate back: the Householder reflection applied
            // on each index.
            bc_matrix_type<NDIM> mom_grad = {};
            for (size_t a = 0; a < NDIM; ++a)
            {
                for (size_t b = 0; b < NDIM; ++b)
                {
                    mom_grad[a][b] = so1c(icl, 1 + a, b);
                }
            }
            bc_matrix_type<NDIM> uv = bc_matmat<NDIM>(bc_matmat<NDIM>(mat, mom_grad), matinv);
            for (size_t it = 1; it < NDIM; ++it)
            {
                uv[0][it] = -uv[0][it];
                uv[it][0] = -uv[it][0];
            }
            bc_matrix_type<NDIM> const out = bc_matmat<NDIM>(bc_matmat<NDIM>(matinv, uv), mat);
            for (size_t a = 0; a < NDIM; ++a)
            {
                for (size_t b = 0; b < NDIM; ++b)
                {
                    so1n(jcl, 1 + a, b) = out[a][b];
                }
            }
        }
    }
}

} /* end namespace detail */

void EulerCore::add_bc(
    EulerBC kind,
    std::vector<int_type> const & faces,
    std::vector<real_type> const & value)
{
    auto const & msh = *m_mesh;
    auto const nface = static_cast<int_type>(msh.nface());
    for (int_type const ifc : faces)
    {
        if (ifc < 0 || ifc >= nface)
        {
            throw std::invalid_argument(std::format(
                "EulerCore::add_bc: face {} out of range [0, {})", ifc, nface));
        }
        // A boundary face is the only kind with a ghost (negative) neighbor.
        if (msh.fccls(ifc, 1) >= 0)
        {
            throw std::invalid_argument(std::format(
                "EulerCore::add_bc: face {} is not a boundary face", ifc));
        }
    }
    if (EulerBC::Inlet == kind)
    {
        size_t const expect = static_cast<size_t>(m_ndim) + 3;
        if (value.size() != expect)
        {
            throw std::invalid_argument(std::format(
                "EulerCore::add_bc: inlet value size {} must equal ndim+3 = {}",
                value.size(),
                expect));
        }
        // value = [rho, v(ndim), p, gamma]; guard the same physical bounds as
        // init_solution so bc_soln stays finite.
        real_type const rho = value[0];
        real_type const p = value[1 + m_ndim];
        real_type const gamma = value[2 + m_ndim];
        if (gamma <= 1.0)
        {
            throw std::invalid_argument(std::format(
                "EulerCore::add_bc: inlet gamma {} must be > 1", gamma));
        }
        if (rho <= 0.0)
        {
            throw std::invalid_argument(std::format(
                "EulerCore::add_bc: inlet rho {} must be > 0", rho));
        }
        if (p < 0.0)
        {
            throw std::invalid_argument(std::format(
                "EulerCore::add_bc: inlet pressure {} must be >= 0", p));
        }
    }
    // Store the face and value lists as SimpleCollector (STYLE.md prefers it
    // over std::vector for fundamental value_type member data).
    EulerBoundary bnd;
    bnd.kind = kind;
    bnd.faces = SimpleCollector<int_type>(faces.size());
    for (size_t i = 0; i < faces.size(); ++i)
    {
        bnd.faces[i] = faces[i];
    }
    bnd.value = SimpleCollector<real_type>(value.size());
    for (size_t i = 0; i < value.size(); ++i)
    {
        bnd.value[i] = value[i];
    }
    m_boundaries.push_back(std::move(bnd));
}

std::vector<std::vector<EulerCore::real_type>>
EulerCore::get_normal_matrix(int_type ifc) const
{
    size_t const ndim = m_ndim;
    std::vector<std::vector<real_type>> out(ndim, std::vector<real_type>(ndim));
    if (2 == m_ndim)
    {
        detail::bc_matrix_type<2> const mat = detail::normal_matrix<2>(detail::face_normal<2>(*m_mesh, ifc));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                out[i][j] = mat[i][j];
            }
        }
    }
    else if (3 == m_ndim)
    {
        detail::bc_matrix_type<3> const mat = detail::normal_matrix<3>(detail::face_normal<3>(*m_mesh, ifc));
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                out[i][j] = mat[i][j];
            }
        }
    }
    else
    {
        throw std::invalid_argument("EulerCore::get_normal_matrix: ndim must be 2 or 3");
    }
    return out;
}

void EulerCore::bc_soln()
{
    if (2 == m_ndim)
    {
        detail::bc_soln_impl<2>(*this);
    }
    else if (3 == m_ndim)
    {
        detail::bc_soln_impl<3>(*this);
    }
    else
    {
        throw std::invalid_argument("EulerCore::bc_soln: ndim must be 2 or 3");
    }
}

void EulerCore::bc_dsoln()
{
    if (2 == m_ndim)
    {
        detail::bc_dsoln_impl<2>(*this);
    }
    else if (3 == m_ndim)
    {
        detail::bc_dsoln_impl<3>(*this);
    }
    else
    {
        throw std::invalid_argument("EulerCore::bc_dsoln: ndim must be 2 or 3");
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
