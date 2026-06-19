/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RStaticMesh.hpp> // Must be the first include.

#include <solvcon/pilot/common_detail.hpp>

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

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
