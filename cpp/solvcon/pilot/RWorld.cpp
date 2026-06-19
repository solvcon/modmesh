/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RWorld.hpp> // Must be the first include.

#include <QTechnique>
#include <QPointSize>
#include <Qt3DExtras/QPerVertexColorMaterial>

#include <limits>
#include <stdexcept>

#include <solvcon/pilot/common_detail.hpp>

namespace solvcon
{

RVertices::RVertices(std::shared_ptr<WorldFp64> const & world, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_geometry(new Qt3DCore::QGeometry())
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
{
    size_t const npoint = world->npoint();

    if (npoint > 0)
    {
        {
            // Build the Qt node (vertex) coordinate buffer.
            auto * vertices = new Qt3DCore::QAttribute(m_geometry);

            vertices->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
            vertices->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
            vertices->setVertexBaseType(Qt3DCore::QAttribute::Float);
            vertices->setVertexSize(3);

            QVector3D min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            QVector3D max_pt(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

            auto * buf = new Qt3DCore::QBuffer(m_geometry);
            {
                QByteArray barray;
                barray.resize(npoint * 3 * sizeof(float));
                SimpleArray<float> sarr = makeSimpleArray<float>(barray, small_vector<size_t>{npoint, 3}, /*view*/ true);
                for (size_t i = 0; i < world->npoint(); ++i)
                {
                    Point3dFp64 const & v = world->point(i);
                    sarr(i, 0) = v[0];
                    sarr(i, 1) = v[1];
                    sarr(i, 2) = v[2];

                    min_pt.setX(std::min(min_pt.x(), sarr(i, 0)));
                    min_pt.setY(std::min(min_pt.y(), sarr(i, 1)));
                    min_pt.setZ(std::min(min_pt.z(), sarr(i, 2)));
                    max_pt.setX(std::max(max_pt.x(), sarr(i, 0)));
                    max_pt.setY(std::max(max_pt.y(), sarr(i, 1)));
                    max_pt.setZ(std::max(max_pt.z(), sarr(i, 2)));
                }
                buf->setData(barray);
            }
            vertices->setBuffer(buf);
            vertices->setByteStride(3 * sizeof(float));
            vertices->setCount(npoint);

            m_geometry->addAttribute(vertices);

            RScene * parent = qobject_cast<RScene *>(this->parent());
            if (parent)
            {
                parent->updateBoundingBox(min_pt, max_pt);
            }
        }

        {
            // Build the Qt node index buffer.
            auto * indices = new Qt3DCore::QAttribute(m_geometry);

            indices->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
            indices->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);

            auto * buf = new Qt3DCore::QBuffer(m_geometry);
            {
                QByteArray barray;
                barray.resize(npoint * sizeof(uint32_t));
                SimpleArray<uint32_t> sarr = makeSimpleArray<uint32_t>(barray, small_vector<size_t>{npoint}, /*view*/ true);
                for (size_t i = 0; i < world->npoint(); ++i)
                {
                    sarr(i) = i;
                }
                buf->setData(barray);
            }
            indices->setBuffer(buf);
            indices->setCount(npoint);

            m_geometry->addAttribute(indices);
        }

        m_renderer->setGeometry(m_geometry);
        m_renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Points);

        addComponent(m_renderer);

        // Update material
        {
            auto effect = m_material->effect();
            for (Qt3DRender::QTechnique * t : effect->techniques())
            {
                for (Qt3DRender::QRenderPass * rp : t->renderPasses())
                {
                    auto ps = new Qt3DRender::QPointSize(m_material);
                    ps->setSizeMode(Qt3DRender::QPointSize::SizeMode::Fixed);
                    ps->setValue(4.0f);
                    rp->addRenderState(ps);
                }
            }
        }

        addComponent(m_material);
    }
}

RLines::RLines(std::shared_ptr<WorldFp64> const & world, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_geometry(new Qt3DCore::QGeometry())
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
{
    // Collect all segments except those from removed shapes.
    std::shared_ptr<SegmentPadFp64> segments = world->collect_live_segments();
    // Create sampled segments in a pad from the live curves (skipping DEAD shapes).
    std::shared_ptr<SegmentPadFp64> csegs = world->collect_live_curves()->sample(/*length*/ 0.1);
    // Extend the overall segment pad with the sampled segments
    segments->extend_with(*csegs);
    // Number of points is twice of that of segments
    size_t npoint = segments->size() * 2;

    /*
     * Fence the geometry building code to prevent the exception from Qt:
     * "QByteArray size disagrees with the requested shape"
     */
    if (npoint > 0)
    {
        {
            // Build the Qt node (vertex) coordinate buffer.
            auto * vertices = new Qt3DCore::QAttribute(m_geometry);

            vertices->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
            vertices->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
            vertices->setVertexBaseType(Qt3DCore::QAttribute::Float);
            vertices->setVertexSize(3);

            QVector3D min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            QVector3D max_pt(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

            auto * buf = new Qt3DCore::QBuffer(m_geometry);
            {
                QByteArray barray;
                barray.resize(npoint * 3 * sizeof(float));
                SimpleArray<float> sarr = makeSimpleArray<float>(barray, small_vector<size_t>{npoint, 3}, /*view*/ true);
                size_t ipt = 0;
                for (size_t i = 0; i < segments->size(); ++i)
                {
                    Segment3dFp64 const & s = segments->get(i);
                    sarr(ipt, 0) = s.p0()[0];
                    sarr(ipt, 1) = s.p0()[1];
                    sarr(ipt, 2) = s.p0()[2];
                    min_pt.setX(std::min(min_pt.x(), sarr(ipt, 0)));
                    min_pt.setY(std::min(min_pt.y(), sarr(ipt, 1)));
                    min_pt.setZ(std::min(min_pt.z(), sarr(ipt, 2)));
                    max_pt.setX(std::max(max_pt.x(), sarr(ipt, 0)));
                    max_pt.setY(std::max(max_pt.y(), sarr(ipt, 1)));
                    max_pt.setZ(std::max(max_pt.z(), sarr(ipt, 2)));
                    ++ipt;

                    sarr(ipt, 0) = s.p1()[0];
                    sarr(ipt, 1) = s.p1()[1];
                    sarr(ipt, 2) = s.p1()[2];
                    min_pt.setX(std::min(min_pt.x(), sarr(ipt, 0)));
                    min_pt.setY(std::min(min_pt.y(), sarr(ipt, 1)));
                    min_pt.setZ(std::min(min_pt.z(), sarr(ipt, 2)));
                    max_pt.setX(std::max(max_pt.x(), sarr(ipt, 0)));
                    max_pt.setY(std::max(max_pt.y(), sarr(ipt, 1)));
                    max_pt.setZ(std::max(max_pt.z(), sarr(ipt, 2)));
                    ++ipt;
                }
                buf->setData(barray);
            }
            vertices->setBuffer(buf);
            vertices->setByteStride(3 * sizeof(float));
            vertices->setCount(npoint);

            m_geometry->addAttribute(vertices);

            RScene * parent = qobject_cast<RScene *>(this->parent());
            if (parent)
            {
                parent->updateBoundingBox(min_pt, max_pt);
            }
        }

        {
            // Build the Qt node index buffer.
            auto * indices = new Qt3DCore::QAttribute(m_geometry);

            indices->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
            indices->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);

            size_t nedge = segments->size();

            auto * buf = new Qt3DCore::QBuffer(m_geometry);
            {
                QByteArray barray;
                barray.resize(nedge * 2 * sizeof(uint32_t));
                SimpleArray<uint32_t> sarr = makeSimpleArray<uint32_t>(barray, small_vector<size_t>{nedge, 2}, /*view*/ true);
                size_t ied = 0;
                size_t ipt = 0;
                for (size_t i = 0; i < segments->size(); ++i)
                {
                    sarr(ied, 0) = ipt++;
                    sarr(ied, 1) = ipt++;
                    ++ied;
                }
                buf->setData(barray);
            }
            indices->setBuffer(buf);
            indices->setCount(nedge * 2);

            m_geometry->addAttribute(indices);
        }

        m_renderer->setGeometry(m_geometry);
        m_renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);

        addComponent(m_renderer);

        addComponent(m_material);
    }
}

RColorField::RColorField(
    SimpleArray<float> const & vertices,
    SimpleArray<float> const & colors,
    SimpleArray<uint32_t> const & indices,
    Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_geometry(new Qt3DCore::QGeometry(this))
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QPerVertexColorMaterial())
{
    // Require (nvert, 3) vertices, a matching (nvert, 3) color table, and
    // (ntri, 3) triangle indices; mismatches would feed Qt malformed buffers.
    if (vertices.ndim() != 2 || vertices.shape(1) != 3)
    {
        throw std::invalid_argument("RColorField: vertices must have shape (nvert, 3)");
    }
    if (colors.ndim() != 2 || colors.shape(0) != vertices.shape(0) || colors.shape(1) != 3)
    {
        throw std::invalid_argument("RColorField: colors must have shape (nvert, 3) matching vertices");
    }
    if (indices.ndim() != 2 || indices.shape(1) != 3)
    {
        throw std::invalid_argument("RColorField: indices must have shape (ntri, 3)");
    }

    size_t const nvert = vertices.shape(0);
    size_t const ntri = indices.shape(0);

    /*
     * Fence the geometry building to skip empty input, mirroring RLines: Qt
     * throws "QByteArray size disagrees with the requested shape" on a
     * zero-sized buffer.
     */
    if (nvert == 0 || ntri == 0)
    {
        return;
    }

    {
        // Vertex coordinate buffer; also accumulate the bounding box.
        auto * attr = new Qt3DCore::QAttribute(m_geometry);
        attr->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
        attr->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        attr->setVertexBaseType(Qt3DCore::QAttribute::Float);
        attr->setVertexSize(3);

        QVector3D min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        QVector3D max_pt(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

        auto * buf = new Qt3DCore::QBuffer(m_geometry);
        {
            QByteArray barray;
            barray.resize(nvert * 3 * sizeof(float));
            SimpleArray<float> sarr = makeSimpleArray<float>(barray, small_vector<size_t>{nvert, 3}, /*view*/ true);
            for (size_t i = 0; i < nvert; ++i)
            {
                for (size_t d = 0; d < 3; ++d)
                {
                    sarr(i, d) = vertices(i, d);
                }
                min_pt.setX(std::min(min_pt.x(), sarr(i, 0)));
                min_pt.setY(std::min(min_pt.y(), sarr(i, 1)));
                min_pt.setZ(std::min(min_pt.z(), sarr(i, 2)));
                max_pt.setX(std::max(max_pt.x(), sarr(i, 0)));
                max_pt.setY(std::max(max_pt.y(), sarr(i, 1)));
                max_pt.setZ(std::max(max_pt.z(), sarr(i, 2)));
            }
            buf->setData(barray);
        }
        attr->setBuffer(buf);
        attr->setByteStride(3 * sizeof(float));
        attr->setCount(nvert);
        m_geometry->addAttribute(attr);

        RScene * scene = qobject_cast<RScene *>(parent);
        if (scene)
        {
            scene->updateBoundingBox(min_pt, max_pt);
        }
    }

    {
        // Per-vertex color buffer, read by the per-vertex-color material.
        auto * attr = new Qt3DCore::QAttribute(m_geometry);
        attr->setName(Qt3DCore::QAttribute::defaultColorAttributeName());
        attr->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        attr->setVertexBaseType(Qt3DCore::QAttribute::Float);
        attr->setVertexSize(3);

        auto * buf = new Qt3DCore::QBuffer(m_geometry);
        {
            QByteArray barray;
            barray.resize(nvert * 3 * sizeof(float));
            SimpleArray<float> sarr = makeSimpleArray<float>(barray, small_vector<size_t>{nvert, 3}, /*view*/ true);
            for (size_t i = 0; i < nvert; ++i)
            {
                for (size_t d = 0; d < 3; ++d)
                {
                    sarr(i, d) = colors(i, d);
                }
            }
            buf->setData(barray);
        }
        attr->setBuffer(buf);
        attr->setByteStride(3 * sizeof(float));
        attr->setCount(nvert);
        m_geometry->addAttribute(attr);
    }

    {
        // Triangle index buffer.
        auto * attr = new Qt3DCore::QAttribute(m_geometry);
        attr->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
        attr->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);

        auto * buf = new Qt3DCore::QBuffer(m_geometry);
        {
            QByteArray barray;
            barray.resize(ntri * 3 * sizeof(uint32_t));
            SimpleArray<uint32_t> sarr = makeSimpleArray<uint32_t>(barray, small_vector<size_t>{ntri, 3}, /*view*/ true);
            for (size_t i = 0; i < ntri; ++i)
            {
                for (size_t d = 0; d < 3; ++d)
                {
                    sarr(i, d) = indices(i, d);
                }
            }
            buf->setData(barray);
        }
        attr->setBuffer(buf);
        attr->setCount(ntri * 3);
        m_geometry->addAttribute(attr);
    }

    m_renderer->setGeometry(m_geometry);
    m_renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
    addComponent(m_renderer);
    addComponent(m_material);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
