/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RStaticMesh.hpp> // Must be the first include.

#include <solvcon/pilot/common_detail.hpp>

#include <Qt3DExtras/QPerVertexColorMaterial>

#include <QCullFace>
#include <QEffect>
#include <QRenderPass>
#include <QTechnique>

#include <array>
#include <cmath>
#include <limits>

namespace solvcon
{

RStaticMesh::RStaticMesh(std::shared_ptr<StaticMesh> const & static_mesh, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_geometry(new Qt3DCore::QGeometry(this))
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
{
    update_geometry(*static_mesh);
    m_renderer->setGeometry(m_geometry);
    m_renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
    addComponent(m_renderer);
    addComponent(m_material);
}

void RStaticMesh::update_geometry_impl(StaticMesh const & mh, Qt3DCore::QGeometry * geom)
{
    {
        // Build the Qt node (vertex) coordinate buffer.
        auto * vertices = new Qt3DCore::QAttribute(geom);

        vertices->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
        vertices->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        vertices->setVertexBaseType(Qt3DCore::QAttribute::Float);
        vertices->setVertexSize(3);
        auto * buf = new Qt3DCore::QBuffer(geom);
        {
            // Copy mesh node coordinates into the Qt buffer.
            QByteArray barray;
            barray.resize(mh.nnode() * 3 * sizeof(float));
            SimpleArray<float> sarr = makeSimpleArray<float>(barray, small_vector<size_t>{mh.nnode(), 3}, /*view*/ true);
            for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
            {
                sarr(ind, 0) = mh.ndcrd(ind, 0);
                sarr(ind, 1) = mh.ndcrd(ind, 1);
                sarr(ind, 2) = (3 == mh.ndim()) ? mh.ndcrd(ind, 2) : 0;
            }
            buf->setData(barray);
        }
        vertices->setBuffer(buf);
        vertices->setByteStride(3 * sizeof(float));
        vertices->setCount(mh.nnode());

        geom->addAttribute(vertices);
    }

    {
        // Build the Qt node index buffer.
        auto * indices = new Qt3DCore::QAttribute(geom);

        indices->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
        indices->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
        auto * buf = new Qt3DCore::QBuffer(geom);
        buf->setData(makeQByteArray(mh.ednds()));
        indices->setBuffer(buf);
        indices->setCount(mh.nedge() * 2);

        geom->addAttribute(indices);
    }
}

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

RBoundary::RBoundary(std::shared_ptr<StaticMesh> const & mesh, int ibc, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_ibc(ibc)
    , m_geometry(new Qt3DCore::QGeometry(this))
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QPerVertexColorMaterial())
{
    build_geometry(*mesh, ibc, m_geometry);
    // build_geometry leaves no attribute when the set has no face; an empty
    // buffer would make Qt throw. addComponent would otherwise adopt these, so
    // free them here before bailing out.
    if (m_geometry->attributes().isEmpty())
    {
        delete m_renderer;
        m_renderer = nullptr;
        delete m_material;
        m_material = nullptr;
        return;
    }
    m_renderer->setGeometry(m_geometry);
    m_renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
    addComponent(m_renderer);
    // The ribbon is a flat two-sided strip whose quads are wound either way,
    // so disable face culling to keep every edge visible.
    for (Qt3DRender::QTechnique * tech : m_material->effect()->techniques())
    {
        for (Qt3DRender::QRenderPass * pass : tech->renderPasses())
        {
            auto * cull = new Qt3DRender::QCullFace(m_material);
            cull->setMode(Qt3DRender::QCullFace::NoCulling);
            pass->addRenderState(cull);
        }
    }
    addComponent(m_material);
}

void RBoundary::build_geometry(StaticMesh const & mh, int ibc, Qt3DCore::QGeometry * geom)
{
    // Gather the boundary-set edges as node-index pairs; with none there is
    // nothing to draw and the geometry is left empty.
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

    // glLineWidth is unreliable (clamped to 1 in the OpenGL core profile), so
    // the thickness is built from triangles instead. The half-width is a small
    // fraction of the mesh extent, so the ribbon reads about twice the
    // hairline wireframe and stays sensible at any zoom.
    float lo[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    float hi[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    auto node = [&mh](uint32_t ind, size_t dim) -> float
    { return (dim < mh.ndim()) ? static_cast<float>(mh.ndcrd(ind, dim)) : 0.0f; };
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

    // Expand each edge into a flat quad (two triangles) offset perpendicular to
    // the edge in the view plane; the meshes are 2D, so the cross product with
    // the z axis gives the in-plane normal.
    std::array<float, 3> const color = boundary_color(ibc);
    QByteArray vbuf;
    QByteArray cbuf;
    QByteArray ibuf;
    vbuf.resize(nedge * 4 * 3 * sizeof(float));
    cbuf.resize(nedge * 4 * 3 * sizeof(float));
    ibuf.resize(nedge * 6 * sizeof(uint32_t));
    SimpleArray<float> verts = makeSimpleArray<float>(vbuf, small_vector<size_t>{nedge * 4, 3}, /*view*/ true);
    SimpleArray<float> colors = makeSimpleArray<float>(cbuf, small_vector<size_t>{nedge * 4, 3}, /*view*/ true);
    SimpleArray<uint32_t> idx = makeSimpleArray<uint32_t>(ibuf, small_vector<size_t>{nedge * 6}, /*view*/ true);
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
        size_t const base = ie * 4;
        for (size_t ic = 0; ic < 4; ++ic)
        {
            verts(base + ic, 0) = px[ic];
            verts(base + ic, 1) = py[ic];
            verts(base + ic, 2) = pz[ic];
            colors(base + ic, 0) = color[0];
            colors(base + ic, 1) = color[1];
            colors(base + ic, 2) = color[2];
        }
        size_t const iib = ie * 6;
        idx(iib + 0) = static_cast<uint32_t>(base + 0);
        idx(iib + 1) = static_cast<uint32_t>(base + 1);
        idx(iib + 2) = static_cast<uint32_t>(base + 2);
        idx(iib + 3) = static_cast<uint32_t>(base + 0);
        idx(iib + 4) = static_cast<uint32_t>(base + 2);
        idx(iib + 5) = static_cast<uint32_t>(base + 3);
    }

    {
        // Ribbon vertex coordinates.
        auto * attr = new Qt3DCore::QAttribute(geom);
        attr->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
        attr->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        attr->setVertexBaseType(Qt3DCore::QAttribute::Float);
        attr->setVertexSize(3);
        auto * buf = new Qt3DCore::QBuffer(geom);
        buf->setData(vbuf);
        attr->setBuffer(buf);
        attr->setByteStride(3 * sizeof(float));
        attr->setCount(nedge * 4);
        geom->addAttribute(attr);
    }

    {
        // A flat color per vertex feeds the per-vertex-color material.
        auto * attr = new Qt3DCore::QAttribute(geom);
        attr->setName(Qt3DCore::QAttribute::defaultColorAttributeName());
        attr->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        attr->setVertexBaseType(Qt3DCore::QAttribute::Float);
        attr->setVertexSize(3);
        auto * buf = new Qt3DCore::QBuffer(geom);
        buf->setData(cbuf);
        attr->setBuffer(buf);
        attr->setByteStride(3 * sizeof(float));
        attr->setCount(nedge * 4);
        geom->addAttribute(attr);
    }

    {
        // Triangle index buffer, two triangles per edge.
        auto * attr = new Qt3DCore::QAttribute(geom);
        attr->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
        attr->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
        auto * buf = new Qt3DCore::QBuffer(geom);
        buf->setData(ibuf);
        attr->setBuffer(buf);
        attr->setCount(static_cast<uint32_t>(nedge * 6));
        geom->addAttribute(attr);
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
