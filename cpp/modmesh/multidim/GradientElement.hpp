#pragma once

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

/**
 * Gradient element geometry for the CESE dual mesh.
 * Port of solvcon GradientElement.hpp.
 */

#include <modmesh/mesh/mesh.hpp>

#include <array>

namespace modmesh
{

struct GradientElementType
{

    static constexpr size_t NFGE_MAX = 8;
    static constexpr size_t FGENFC_MAX = 3;

    using face_list_type = std::array<int32_t, FGENFC_MAX>;

    int32_t clnfc = -1;
    int32_t nfge = -1;
    // 1-based indices into the per-cell gradient-eval-point array.
    std::array<face_list_type, NFGE_MAX> faces = {};

}; /* end struct GradientElementType */

namespace detail
{

class GradientElementTypeGroup
{

public:

    GradientElementType const & operator[](size_t id) const { return m_types[id]; }

    static GradientElementTypeGroup const & get_instance()
    {
        static GradientElementTypeGroup const inst;
        return inst;
    }

private:

    GradientElementTypeGroup();

    std::array<GradientElementType, CellType::NTYPE + 1> m_types;

}; /* end class GradientElementTypeGroup */

} /* end namespace detail */

inline GradientElementType const & getype(size_t id)
{
    return detail::GradientElementTypeGroup::get_instance()[id];
}

class GradientElement
{

public:

    using int_type = int32_t;
    using real_type = double;

    GradientElement(
        StaticMesh const & mesh,
        SimpleArray<real_type> const & cecnd,
        int_type icl,
        real_type tau);

    int_type icl() const { return m_icl; }
    uint8_t ndim() const { return m_ndim; }
    int_type clnfc() const { return m_clnfc; }
    int_type rcl(int_type ifl) const { return m_rcls[ifl]; }
    real_type idis(int_type ifl, int_type d) const { return m_idis[ifl][d]; }
    real_type jdis(int_type ifl, int_type d) const { return m_jdis[ifl][d]; }

private:

    // Index of the self cell that owns this gradient element.
    int_type m_icl;
    // Number of spatial dimensions (2 or 3).
    uint8_t m_ndim;
    // Number of cell faces, i.e. the number of gradient evaluation points.
    // This is the valid extent of the per-face axis (axis 0) of the arrays
    // below and never exceeds StaticMesh::CLMFC.
    int_type m_clnfc;
    // Neighbor cell across each face.  Axis 0: face index ifl, valid in [0,
    // clnfc), capacity CLMFC.  A negative entry denotes a ghost cell.
    std::array<int_type, StaticMesh::CLMFC> m_rcls;
    // Displacement from the self solution point (the self cell's CE centroid)
    // to the gradient evaluation point of each face.  Axis 0: face index ifl,
    // valid in [0, clnfc), capacity CLMFC.  Axis 1: dimension d, valid in [0,
    // ndim), capacity 3 (to hold up to 3D).
    std::array<std::array<real_type, 3>, StaticMesh::CLMFC> m_idis;
    // Displacement from the neighboring solution point (the neighbor cell's CE
    // centroid) to the same gradient evaluation point.  Same axes and extents
    // as m_idis.
    std::array<std::array<real_type, 3>, StaticMesh::CLMFC> m_jdis;

}; /* end class GradientElement */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
