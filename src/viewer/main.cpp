/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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

#include <algorithm>

#include <QGuiApplication>
#include <QByteArray>

#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QEntity>
#include <Qt3DCore/QGeometry>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QTransform>

#include <Qt3DRender/QCamera>

#include <Qt3DExtras/QDiffuseSpecularMaterial>
#include <Qt3DExtras/QCuboidMesh>

#include <QOrbitCameraController>

#include <qt3dwindow.h>

#include <modmesh/modmesh.hpp>

std::shared_ptr<modmesh::StaticMesh2d> make_sample_static_mesh()
{
    auto mh = modmesh::StaticMesh2d::construct(/*nnode*/ 4, /*nface*/ 0, /*ncell*/ 3);

    mh->ndcrd(0, 0) = 0;
    mh->ndcrd(0, 1) = 0;
    mh->ndcrd(1, 0) = -1;
    mh->ndcrd(1, 1) = -1;
    mh->ndcrd(2, 0) = 1;
    mh->ndcrd(2, 1) = -1;
    mh->ndcrd(3, 0) = 0;
    mh->ndcrd(3, 1) = 1;

    std::fill(mh->cltpn().begin(), mh->cltpn().end(), modmesh::CellType::TRIANGLE);

    mh->clnds(0, 0) = 3;
    mh->clnds(0, 1) = 0;
    mh->clnds(0, 2) = 1;
    mh->clnds(0, 3) = 2;
    mh->clnds(1, 0) = 3;
    mh->clnds(1, 1) = 0;
    mh->clnds(1, 2) = 2;
    mh->clnds(1, 3) = 3;
    mh->clnds(2, 0) = 3;
    mh->clnds(2, 1) = 0;
    mh->clnds(2, 2) = 3;
    mh->clnds(2, 3) = 1;

    mh->build_interior(/*do_metric*/ true);
    mh->build_boundary();
    mh->build_ghost();

    return mh;
}

void draw_mesh(modmesh::StaticMesh2d const & mh, Qt3DCore::QEntity * root)
{
    auto * geometry = new Qt3DCore::QGeometry(root);

    {
        auto * buf = new Qt3DCore::QBuffer(geometry);
        {
            QByteArray barray;
            barray.resize(3 * mh.nnode() * sizeof(float));
            float * ptr = reinterpret_cast<float *>(barray.data());
            for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
            {
                *ptr++ = mh.ndcrd(ind, 0);
                *ptr++ = mh.ndcrd(ind, 1);
                *ptr++ = 0;
            }
            buf->setData(barray);
        }

        auto * attr = new Qt3DCore::QAttribute(geometry);
        attr->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
        attr->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        attr->setVertexBaseType(Qt3DCore::QAttribute::Float);
        attr->setVertexSize(3);
        attr->setBuffer(buf);
        attr->setByteStride(3 * sizeof(float));
        attr->setCount(mh.nnode());

        geometry->addAttribute(attr); // We add the vertices in the geometry
    }

    {
        uint32_t nend = 0;
        for (uint32_t ifc = 0; ifc < mh.nface(); ++ifc)
        {
            nend += mh.fcnds(ifc, 0);
        }

        auto * buf = new Qt3DCore::QBuffer(geometry);
        {
            QByteArray barray;
            barray.resize(nend * sizeof(uint32_t));
            auto * indices = reinterpret_cast<uint32_t *>(barray.data());
            for (uint32_t ifc = 0; ifc < mh.nface(); ++ifc)
            {
                for (int32_t inf = 1; inf <= mh.fcnds(ifc, 0); ++inf)
                {
                    *indices++ = mh.fcnds(ifc, inf);
                }
            }
            buf->setData(barray);
        }

        auto * attr = new Qt3DCore::QAttribute(geometry);
        attr->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
        attr->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
        attr->setBuffer(buf);
        attr->setCount(nend);

        geometry->addAttribute(attr);
    }

    auto * line_entity = new Qt3DCore::QEntity(root);
    {
        auto * line_render = new Qt3DRender::QGeometryRenderer(root);
        line_render->setGeometry(geometry);
        line_render->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
        line_entity->addComponent(line_render);

        Qt3DRender::QMaterial * line_material = new Qt3DExtras::QDiffuseSpecularMaterial(root);
        line_entity->addComponent(line_material);
    }
}

Qt3DCore::QEntity * create_scene(modmesh::StaticMesh2d const & mh)
{
    // Root entity.
    Qt3DCore::QEntity * root = new Qt3DCore::QEntity;

    draw_mesh(mh, root);

    return root;
}

int main(int argc, char * argv[])
{
    QGuiApplication app(argc, argv);
    Qt3DExtras::Qt3DWindow view;

    auto mh = make_sample_static_mesh();
    Qt3DCore::QEntity * scene = create_scene(*mh);

    // Camera.
    Qt3DRender::QCamera * camera = view.camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(0, 0, 40.0f));
    camera->setViewCenter(QVector3D(0, 0, 0));

    Qt3DExtras::QOrbitCameraController * com_control = new Qt3DExtras::QOrbitCameraController(scene);
    com_control->setLinearSpeed(50.0f);
    com_control->setLookSpeed(180.0f);
    com_control->setCamera(camera);

    view.setRootEntity(scene);
    view.show();

    return app.exec();
}
