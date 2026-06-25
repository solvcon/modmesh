/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RMeshBoundary.hpp> // Must be the first include.

#include <array>
#include <cmath>
#include <limits>

namespace solvcon
{

namespace
{

/// Pick a distinct, saturated color for a boundary set so neighboring sets
/// stay tellable apart; the palette repeats for meshes with many sets.
std::array<float, 3> boundary_color(int ibc)
{
    static const std::array<std::array<float, 3>, 6> palette{{
        {1.0f, 0.20f, 0.20f}, // red
        {0.20f, 0.60f, 1.0f}, // blue
        {0.20f, 0.80f, 0.30f}, // green
        {1.0f, 0.70f, 0.10f}, // amber
        {0.80f, 0.30f, 0.90f}, // purple
        {0.10f, 0.80f, 0.80f}, // teal
    }};
    return palette.at(static_cast<size_t>(ibc < 0 ? 0 : ibc) % palette.size());
}

} /* end namespace */

RMeshBoundary::RMeshBoundary(std::shared_ptr<StaticMesh> const & mesh, int ibc)
    : m_ibc(ibc)
{
    build(*mesh, ibc);
}

void RMeshBoundary::build(StaticMesh const & mh, int ibc)
{
    // Gather the boundary-set edges as node-index pairs; with none there is
    // nothing to draw.
    SimpleCollector<uint32_t> ends;
    SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
    for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
    {
        if (bndfcs(ibnd, 1) != ibc)
        {
            continue;
        }
        int32_t const ifc = bndfcs(ibnd, 0);
        int32_t const nnd = mh.fcnds(ifc, 0);
        // A 2D face is a single edge; a 3D face is a polygon whose rim edges
        // close back to the first node.
        for (int32_t ind = 1; ind <= nnd; ++ind)
        {
            ends.push_back(static_cast<uint32_t>(mh.fcnds(ifc, ind)));
            int32_t const next = (ind == nnd) ? 1 : ind + 1;
            ends.push_back(static_cast<uint32_t>(mh.fcnds(ifc, next)));
            if (2 == nnd)
            {
                break; // avoid drawing the same segment twice.
            }
        }
    }
    size_t const nedge = ends.size() / 2;
    if (0 == nedge)
    {
        return;
    }

    auto node = [&mh](uint32_t ind, size_t dim) -> float
    { return (dim < mh.ndim()) ? static_cast<float>(mh.ndcrd(ind, dim)) : 0.0f; };

    // The ribbon is widened in the xy plane and lifted along +z, which is
    // exact for a z-planar (2D) boundary: the case the highlight targets,
    // matching the retired prototype. A genuine out-of-plane 3D boundary edge
    // still draws but is not oriented to its face.
    //
    // The ribbon half-width is a small fraction of the mesh extent, so it
    // reads about twice the hairline wireframe at any zoom.
    float lo[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    float hi[3] = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        for (size_t dim = 0; dim < 3; ++dim)
        {
            lo[dim] = std::min(lo[dim], node(ind, dim));
            hi[dim] = std::max(hi[dim], node(ind, dim));
        }
    }
    float const diag = std::sqrt(
        (hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2]));
    float const half = (diag > 0.0f ? diag : 1.0f) * 0.0035f;
    // Lift the ribbon toward the viewer (the meshes are 2D, so +z) so it wins
    // the depth test against the coplanar wireframe instead of z-fighting it.
    float const lift = half;

    std::array<float, 3> const color = boundary_color(ibc);
    m_interleaved.reserve(nedge * 4 * 6);
    m_indices.reserve(nedge * 6);
    for (size_t ie = 0; ie < nedge; ++ie)
    {
        uint32_t const i0 = ends[ie * 2];
        uint32_t const i1 = ends[ie * 2 + 1];
        float const dx = node(i1, 0) - node(i0, 0);
        float const dy = node(i1, 1) - node(i0, 1);
        float nx = dy;
        float ny = -dx;
        float len = std::sqrt(nx * nx + ny * ny);
        if (len < std::numeric_limits<float>::epsilon())
        {
            nx = 1.0f;
            ny = 0.0f;
            len = 1.0f;
        }
        float const ox = nx / len * half;
        float const oy = ny / len * half;
        // Corner order p0+o, p0-o, p1-o, p1+o makes the two triangles below.
        float const px[4] = {node(i0, 0) + ox, node(i0, 0) - ox, node(i1, 0) - ox, node(i1, 0) + ox};
        float const py[4] = {node(i0, 1) + oy, node(i0, 1) - oy, node(i1, 1) - oy, node(i1, 1) + oy};
        float const pz[4] = {node(i0, 2) + lift, node(i0, 2) + lift, node(i1, 2) + lift, node(i1, 2) + lift};
        uint32_t const base = static_cast<uint32_t>(m_interleaved.size() / 6);
        for (size_t ic = 0; ic < 4; ++ic)
        {
            m_interleaved.push_back(px[ic]);
            m_interleaved.push_back(py[ic]);
            m_interleaved.push_back(pz[ic]);
            m_interleaved.push_back(color[0]);
            m_interleaved.push_back(color[1]);
            m_interleaved.push_back(color[2]);
        }
        m_indices.push_back(base + 0);
        m_indices.push_back(base + 1);
        m_indices.push_back(base + 2);
        m_indices.push_back(base + 0);
        m_indices.push_back(base + 2);
        m_indices.push_back(base + 3);
    }

    setColor(QVector4D(color[0], color[1], color[2], 1.0f));
}

QRhiVertexInputLayout RMeshBoundary::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    layout.setBindings({{6 * sizeof(float)}});
    layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
    });
    return layout;
}

void RMeshBoundary::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
{
    if (0 == m_interleaved.size() || 0 == m_indices.size())
    {
        return;
    }

    quint32 const vbytes = static_cast<quint32>(m_interleaved.size() * sizeof(float));
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, vbytes));
    m_vbuf->create();
    batch->uploadStaticBuffer(m_vbuf.get(), m_interleaved.data());
    m_vertex_count = static_cast<quint32>(m_interleaved.size() / 6);

    quint32 const ibytes = static_cast<quint32>(m_indices.size() * sizeof(uint32_t));
    m_ibuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::IndexBuffer, ibytes));
    m_ibuf->create();
    batch->uploadStaticBuffer(m_ibuf.get(), m_indices.data());
    m_index_count = static_cast<quint32>(m_indices.size());
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
