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

namespace modmesh
{

class REntityBase
{

public:

    REntityBase(Qt3DCore::QEntity * ptr)
        : m_ptr(ptr)
    {
    }

    ~REntityBase()
    {
        if (nullptr != m_ptr)
        {
            // If not managed by a parent, signal for deletion.
            if (nullptr == m_ptr->parent())
            {
                m_ptr->deleteLater();
            }
        }
    }

    void set(Qt3DCore::QEntity * ptr) { m_ptr = ptr; }

    Qt3DCore::QEntity const * ptr() const { return m_ptr; }
    Qt3DCore::QEntity * ptr() { return m_ptr; }

    Qt3DCore::QEntity const * operator->() const { return m_ptr; }
    Qt3DCore::QEntity * operator->() { return m_ptr; }

private:

    Qt3DCore::QEntity * m_ptr = nullptr;

}; /* end class REntityBase */

template <uint8_t ND>
class RStaticMesh
    : public REntityBase
{

public:

    using mesh_type = StaticMesh<ND>;

    RStaticMesh() = delete;
    ~RStaticMesh() = default;

    RStaticMesh(std::shared_ptr<mesh_type> const & static_mesh)
        : REntityBase(new Qt3DCore::QEntity())
        , m_static_mesh(static_mesh)
        , m_geometry(make_geom(*static_mesh, ptr()))
        , m_renderer(make_renderer(m_geometry))
        , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
    {
        ptr()->addComponent(m_renderer);
        ptr()->addComponent(m_material);
    }

    static Qt3DCore::QGeometry * make_geom(mesh_type const & mh, Qt3DCore::QEntity * root)
    {
        auto * geom = new Qt3DCore::QGeometry(root);

        auto * buf = new Qt3DCore::QBuffer(geom);
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

        auto * vertices = new Qt3DCore::QAttribute(geom);
        vertices->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
        vertices->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        vertices->setVertexBaseType(Qt3DCore::QAttribute::Float);
        vertices->setVertexSize(3);
        vertices->setBuffer(buf);
        vertices->setByteStride(3 * sizeof(float));
        vertices->setCount(mh.nnode());

        geom->addAttribute(vertices);

        uint32_t nend = 0;
        for (uint32_t ifc = 0; ifc < mh.nface(); ++ifc)
        {
            nend += mh.fcnds(ifc, 0);
        }

        buf = new Qt3DCore::QBuffer(geom);
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

        auto * indices = new Qt3DCore::QAttribute(geom);
        indices->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
        indices->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
        indices->setBuffer(buf);
        indices->setCount(nend);

        geom->addAttribute(indices);

        return geom;
    }

    static Qt3DRender::QGeometryRenderer * make_renderer(Qt3DCore::QGeometry * geom)
    {
        auto * renderer = new Qt3DRender::QGeometryRenderer();
        renderer->setGeometry(geom);
        renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
        return renderer;
    }

    mesh_type const & static_mesh() const { return *m_static_mesh; }
    mesh_type & static_mesh() { return *m_static_mesh; }

private:

    std::shared_ptr<mesh_type> m_static_mesh = nullptr;

    Qt3DCore::QGeometry * m_geometry = nullptr;
    Qt3DRender::QGeometryRenderer * m_renderer = nullptr;
    Qt3DRender::QMaterial * m_material = nullptr;

}; /* end class RStaticMesh */

class RScene
    : public REntityBase
{

public:

    RScene()
        : REntityBase(new Qt3DCore::QEntity())
        , m_camera_controller(new Qt3DExtras::QOrbitCameraController(ptr()))
    {
    }

    Qt3DExtras::QOrbitCameraController const * camera_controller() const { return m_camera_controller; }
    Qt3DExtras::QOrbitCameraController * camera_controller() { return m_camera_controller; }

private:

    Qt3DExtras::QOrbitCameraController * m_camera_controller = nullptr;

}; /* end class RScene */

} /* end namespace modmesh */

std::shared_ptr<modmesh::StaticMesh2d> make_3triangles()
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

int main(int argc, char ** argv)
{
    /*
     * TODO: Sequence of application startup:
     *   1. Parsing arguments and parameters.
     *   2. Initialize application globals.
     *   3. Initialize GUI globals.
     *   4. Set up GUI windowing.
     */

    using namespace modmesh;

    // Start application with GUI.
    QGuiApplication app(argc, argv);

    // Create and set up main window.
    Qt3DExtras::Qt3DWindow window;

    {
        // Set up the camera.
        Qt3DRender::QCamera * camera = window.camera();
        camera->lens()->setPerspectiveProjection(45.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
        camera->setPosition(QVector3D(0, 0, 40.0f));
        camera->setViewCenter(QVector3D(0, 0, 0));
    }

    // Create and set up the root scene.
    RScene scene;

    {
        // Set up the camera control.
        auto * control = scene.camera_controller();
        control->setCamera(window.camera());
        control->setLinearSpeed(50.0f);
        control->setLookSpeed(180.0f);

        // Set the mesh to the scene.
        RStaticMesh<2> rmh(make_3triangles());
        rmh->setParent(scene.ptr());
    }

    // Show the window.
    window.setRootEntity(scene.ptr());
    window.show();

    return app.exec();
}
