/*
 * Copyright (c) 2026, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/multidim/GradientElement.hpp>

#include <stdexcept>

namespace modmesh
{

namespace detail
{

GradientElementTypeGroup::GradientElementTypeGroup()
    : m_types{}
{
    GradientElementType & q = m_types[CellType::QUADRILATERAL];
    q.clnfc = 4;
    q.nfge = 4;
    q.faces[0] = {1, 2, -1};
    q.faces[1] = {2, 3, -1};
    q.faces[2] = {3, 4, -1};
    q.faces[3] = {4, 1, -1};

    GradientElementType & t = m_types[CellType::TRIANGLE];
    t.clnfc = 3;
    t.nfge = 3;
    t.faces[0] = {1, 2, -1};
    t.faces[1] = {2, 3, -1};
    t.faces[2] = {3, 1, -1};

    GradientElementType & h = m_types[CellType::HEXAHEDRON];
    h.clnfc = 6;
    h.nfge = 8;
    h.faces[0] = {2, 3, 5};
    h.faces[1] = {6, 3, 2};
    h.faces[2] = {4, 3, 6};
    h.faces[3] = {5, 3, 4};
    h.faces[4] = {5, 1, 2};
    h.faces[5] = {2, 1, 6};
    h.faces[6] = {6, 1, 4};
    h.faces[7] = {4, 1, 5};

    GradientElementType & e = m_types[CellType::TETRAHEDRON];
    e.clnfc = 4;
    e.nfge = 4;
    e.faces[0] = {3, 1, 2};
    e.faces[1] = {2, 1, 4};
    e.faces[2] = {4, 1, 3};
    e.faces[3] = {2, 4, 3};

    GradientElementType & p = m_types[CellType::PRISM];
    p.clnfc = 5;
    p.nfge = 6;
    p.faces[0] = {5, 2, 4};
    p.faces[1] = {3, 2, 5};
    p.faces[2] = {4, 2, 3};
    p.faces[3] = {4, 1, 5};
    p.faces[4] = {5, 1, 3};
    p.faces[5] = {3, 1, 4};

    GradientElementType & y = m_types[CellType::PYRAMID];
    y.clnfc = 5;
    y.nfge = 6;
    y.faces[0] = {1, 5, 2};
    y.faces[1] = {2, 5, 3};
    y.faces[2] = {3, 5, 4};
    y.faces[3] = {4, 5, 1};
    y.faces[4] = {1, 3, 4};
    y.faces[5] = {3, 1, 2};
}

std::array<double, 3> calc_gge_centroid(
    GradientElementType const & ge,
    std::array<double, 3> const & avg,
    std::array<std::array<double, 3>, StaticMesh::CLMFC> const & gp,
    size_t ndim)
{
    using vec3 = std::array<double, 3>;
    vec3 cnd = {0, 0, 0};
    double voc = 0.0;
    for (int32_t isub = 0; isub < ge.nfge; ++isub)
    {
        vec3 subcnd = avg;
        std::array<vec3, 3> dst = {};
        for (size_t ivx = 0; ivx < ndim; ++ivx)
        {
            int32_t const fl = ge.faces[isub][ivx] - 1;
            for (size_t d = 0; d < ndim; ++d)
            {
                subcnd[d] += gp[fl][d];
                dst[ivx][d] = gp[fl][d] - avg[d];
            }
        }
        for (size_t d = 0; d < ndim; ++d)
        {
            subcnd[d] /= static_cast<double>(ndim + 1);
        }

        double vob = 0.0;
        if (2 == ndim)
        {
            vob = dst[0][0] * dst[1][1] - dst[0][1] * dst[1][0];
        }
        else
        {
            // clang-format off
            vob = dst[0][0] * (dst[1][1] * dst[2][2] - dst[1][2] * dst[2][1])
                - dst[0][1] * (dst[1][0] * dst[2][2] - dst[1][2] * dst[2][0])
                + dst[0][2] * (dst[1][0] * dst[2][1] - dst[1][1] * dst[2][0]);
            // clang-format on
        }

        voc += vob;
        for (size_t d = 0; d < ndim; ++d)
        {
            cnd[d] += subcnd[d] * vob;
        }
    }
    for (size_t d = 0; d < ndim; ++d)
    {
        cnd[d] /= voc;
    }
    return cnd;
}

} /* end namespace detail */

GradientElement::GradientElement(
    StaticMesh const & mesh,
    SimpleArray<real_type> const & cecnd,
    int_type icl,
    real_type tau)
    : m_icl(icl)
    , m_ndim(mesh.ndim())
    , m_clnfc(mesh.clfcs(icl, 0))
    , m_rcls{}
    , m_idis{}
    , m_jdis{}
{
    if (m_ndim < 2 || m_ndim > 3)
    {
        throw std::invalid_argument("GradientElement: ndim must be 2 or 3");
    }

    size_t const ndim = m_ndim;

    // Self CE centroid.
    std::array<real_type, 3> icecnd = {0, 0, 0};
    for (size_t d = 0; d < ndim; ++d)
    {
        icecnd[d] = cecnd(icl, d);
    }

    // Gradient evaluation points (absolute positions).
    std::array<std::array<real_type, 3>, StaticMesh::CLMFC> gp = {};
    std::array<std::array<real_type, 3>, StaticMesh::CLMFC> jd = {};

    for (int_type ifl = 0; ifl < m_clnfc; ++ifl)
    {
        int_type const ifc = mesh.clfcs(icl, ifl + 1);
        int_type const jcl = mesh.fcrcl(ifc, icl);
        m_rcls[ifl] = jcl;

        size_t const bce_col = static_cast<size_t>(ifl + 1) * ndim;
        for (size_t d = 0; d < ndim; ++d)
        {
            real_type const mid = cecnd(icl, bce_col + d);
            real_type const jce = (jcl >= 0) ? cecnd(jcl, d) : mesh.clcnd(jcl, d);
            gp[ifl][d] = mid + tau * (jce - mid);
            jd[ifl][d] = gp[ifl][d] - jce;
        }
    }

    // Average of gradient evaluation points.
    std::array<real_type, 3> avg = {0, 0, 0};
    for (int_type ifl = 0; ifl < m_clnfc; ++ifl)
    {
        for (size_t d = 0; d < ndim; ++d)
        {
            avg[d] += gp[ifl][d];
        }
    }
    for (size_t d = 0; d < ndim; ++d)
    {
        avg[d] /= m_clnfc;
    }

    // GGE centroid via sub-element triangulation.
    GradientElementType const & ge = getype(mesh.cltpn(icl));
    std::array<real_type, 3> const cnd = detail::calc_gge_centroid(ge, avg, gp, ndim);

    // Shift so the GGE centroid coincides with the self CE
    // centroid (solution point).
    for (int_type ifl = 0; ifl < m_clnfc; ++ifl)
    {
        for (size_t d = 0; d < ndim; ++d)
        {
            m_idis[ifl][d] = gp[ifl][d] - cnd[d];
            m_jdis[ifl][d] = jd[ifl][d] + icecnd[d] - cnd[d];
        }
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
