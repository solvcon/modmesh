/*
 * Copyright (c) 2023, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/pilot/RWorld.hpp> // Must be the first include.

#include <QTechnique>
#include <QPointSize>

#include <modmesh/pilot/common_detail.hpp>

namespace modmesh
{

RVertices::RVertices(std::shared_ptr<WorldFp64> const & world, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_geometry(new Qt3DCore::QGeometry())
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
    , m_bounding_min(QVector3D(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()))
    , m_bounding_max(QVector3D(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()))
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
            setBounding_min(min_pt);
            setBounding_max(max_pt);
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
    , m_bounding_min(QVector3D(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()))
    , m_bounding_max(QVector3D(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()))
{
    // Create segment pad
    std::shared_ptr<SegmentPadFp64> segments = world->segments()->clone();
    // Create sampled segments in a pad from the curves
    std::shared_ptr<SegmentPadFp64> csegs = world->curves()->sample(/*length*/ 0.1);
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

            setBounding_min(min_pt);
            setBounding_max(max_pt);
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

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
