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

#include <modmesh/pilot/common_detail.hpp>

namespace modmesh
{

RWorld::RWorld(std::shared_ptr<WorldFp64> const & world, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_world(world)
    , m_geometry(new Qt3DCore::QGeometry(this))
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
{
    update_geometry();
    m_renderer->setGeometry(m_geometry);
    m_renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
    addComponent(m_renderer);
    addComponent(m_material);
}

void RWorld::update_geometry()
{
    size_t npoint = m_world->nedge() * 2;
    for (size_t i = 0; i < m_world->nbezier(); ++i)
    {
        npoint += m_world->bezier(i).nlocus();
    }

    /* Fence the geometry building code to prevent the exception from Qt:
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

            auto * buf = new Qt3DCore::QBuffer(m_geometry);
            {
                QByteArray barray;
                barray.resize(npoint * 3 * sizeof(float));
                SimpleArray<float> sarr = makeSimpleArray<float>(barray, small_vector<size_t>{npoint, 3}, /*view*/ true);
                size_t ipt = 0;
                for (size_t i = 0; i < m_world->nedge(); i++)
                {
                    Edge3dFp64 const & e = m_world->edge(i);
                    sarr(ipt, 0) = e.v0()[0];
                    sarr(ipt, 1) = e.v0()[1];
                    sarr(ipt, 2) = e.v0()[2];
                    ++ipt;
                    sarr(ipt, 0) = e.v1()[0];
                    sarr(ipt, 1) = e.v1()[1];
                    sarr(ipt, 2) = e.v1()[2];
                    ++ipt;
                }
                for (size_t i = 0; i < m_world->nbezier(); ++i)
                {
                    Bezier3dFp64 const & b = m_world->bezier(i);
                    for (size_t j = 0; j < b.nlocus(); ++j)
                    {
                        Vector3dFp64 const & v = b.locus(j);
                        sarr(ipt, 0) = v[0];
                        sarr(ipt, 1) = v[1];
                        sarr(ipt, 2) = v[2];
                        ++ipt;
                    }
                }
                buf->setData(barray);
            }
            vertices->setBuffer(buf);
            vertices->setByteStride(3 * sizeof(float));
            vertices->setCount(npoint);

            m_geometry->addAttribute(vertices);
        }

        {
            // Build the Qt node index buffer.
            auto * indices = new Qt3DCore::QAttribute(m_geometry);

            indices->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
            indices->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);

            size_t nedge = m_world->nedge();
            {
                for (size_t i = 0; i < m_world->nbezier(); ++i)
                {
                    Bezier3dFp64 const & b = m_world->bezier(i);
                    nedge += b.nlocus() - 1;
                }
            }

            auto * buf = new Qt3DCore::QBuffer(m_geometry);
            {
                QByteArray barray;
                barray.resize(nedge * 2 * sizeof(uint32_t));
                SimpleArray<uint32_t> sarr = makeSimpleArray<uint32_t>(barray, small_vector<size_t>{nedge, 2}, /*view*/ true);
                size_t ied = 0;
                size_t ipt = 0;
                for (size_t i = 0; i < m_world->nedge(); ++i)
                {
                    sarr(ied, 0) = ipt++;
                    sarr(ied, 1) = ipt++;
                    ++ied;
                }
                for (size_t i = 0; i < m_world->nbezier(); ++i)
                {
                    Bezier3dFp64 const & b = m_world->bezier(i);
                    for (size_t j = 0; j < b.nlocus() - 1; ++j)
                    {
                        sarr(ied, 0) = ipt++;
                        sarr(ied, 1) = ipt;
                        ++ied;
                    }
                    ++ipt;
                }
                buf->setData(barray);
            }
            indices->setBuffer(buf);
            indices->setCount(nedge * 2);

            m_geometry->addAttribute(indices);
        }
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
