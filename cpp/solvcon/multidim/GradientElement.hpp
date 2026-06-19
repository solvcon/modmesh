#pragma once

/*
 * Copyright (c) 2018, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * Gradient element geometry for the CESE dual mesh.
 * Port of solvcon GradientElement.hpp.
 */

#include <solvcon/mesh/mesh.hpp>

#include <array>

namespace solvcon
{

struct GradientElementType
{

    static constexpr size_t NFGE_MAX = 8;
    static constexpr size_t FGENFC_MAX = 3;

    using face_list_type = std::array<int32_t, FGENFC_MAX>;

    int32_t clnfc = -1;
    int32_t nfge = -1;
    // Reciprocal of nfge, used as the uniform gradient-weighting baseline.
    double nfge_inverse = 0.0;
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
    // Fixed 3-capacity row-major matrix/vector for the leading ndim x ndim
    // (ndim is 2 or 3) systems of the gradient reconstruction.  Distinct from
    // the dynamic solvcon::small_vector container.
    using ge_vector_type = std::array<real_type, 3>;
    using ge_matrix_type = std::array<ge_vector_type, 3>;

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

    // Number of fundamental gradient elements (sub-simplices) of this cell.
    int_type nfge() const { return m_getype->nfge; }
    real_type nfge_inverse() const { return m_getype->nfge_inverse; }
    // 1-based face indices forming the ifge-th fundamental gradient element.
    // Only the leading ndim entries are valid; the rest are -1 sentinels.
    GradientElementType::face_list_type const & faces(int_type ifge) const
    {
        return m_getype->faces[ifge];
    }
    // Displacement matrix of the ifge-th fundamental gradient element: row ivx
    // is idis of the face faces(ifge)[ivx].  Only the leading ndim x ndim block
    // is meaningful.
    ge_matrix_type displacement_matrix(int_type ifge) const;
    // Reconstruct the ndim-vector gradient of one fundamental gradient element
    // from the supplied solution deltas udf by solving (displacement matrix) *
    // grad = udf, i.e. grad = adjugate(dst) * udf / determinant(dst).
    ge_vector_type solve_gradient(int_type ifge, ge_vector_type const & udf) const;

    // Small dense linear algebra on the leading ndim x ndim block (ndim is 2 or
    // 3); solvcon/linalg only wraps the dynamic BLAS/LAPACK routines, too heavy
    // for these per-cell solves.  Static, as they depend solely on their
    // arguments, which keeps them directly testable.
    static real_type determinant(ge_matrix_type const & a, size_t ndim);
    static ge_matrix_type adjugate(ge_matrix_type const & a, size_t ndim);
    static ge_vector_type multiply(ge_matrix_type const & a, ge_vector_type const & x, size_t ndim);

private:

    // Geometry type table (sub-element face lists) for this cell's type.  Points
    // into a process-wide singleton, so it outlives every GradientElement.
    GradientElementType const * m_getype = nullptr;
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

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
