#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/solvcon.hpp>

#include <Qt>
#include <QWidget>
#include <Qt3DWindow>

#include <QByteArray>
#include <QGeometryRenderer>

#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QEntity>
#include <Qt3DCore/QGeometry>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QTransform>

#include <Qt3DExtras/QDiffuseSpecularMaterial>

namespace solvcon
{

class RStaticMesh
    : public Qt3DCore::QEntity
{

public:

    RStaticMesh(std::shared_ptr<StaticMesh> const & static_mesh, Qt3DCore::QNode * parent = nullptr);

    void update_geometry(StaticMesh const & mh)
    {
        update_geometry_impl(mh, m_geometry);
    }

private:

    static void update_geometry_impl(StaticMesh const & mh, Qt3DCore::QGeometry * geom);

    Qt3DCore::QGeometry * m_geometry = nullptr;
    Qt3DRender::QGeometryRenderer * m_renderer = nullptr;
    Qt3DRender::QMaterial * m_material = nullptr;

}; /* end class RStaticMesh */

/**
 * Render the faces of a single boundary set as thick colored ribbons so the
 * set stands out over the wireframe mesh.  Each edge is widened into two
 * triangles because glLineWidth is clamped to 1 in the OpenGL core profile.
 * One entity is created per highlighted boundary set; @ref ibc identifies
 * which set it draws.
 */
class RBoundary
    : public Qt3DCore::QEntity
{

public:

    RBoundary(std::shared_ptr<StaticMesh> const & mesh, int ibc, Qt3DCore::QNode * parent = nullptr);

    int ibc() const { return m_ibc; }

private:

    static void build_geometry(StaticMesh const & mh, int ibc, Qt3DCore::QGeometry * geom);

    int m_ibc = -1;
    Qt3DCore::QGeometry * m_geometry = nullptr;
    Qt3DRender::QGeometryRenderer * m_renderer = nullptr;
    Qt3DRender::QMaterial * m_material = nullptr;

}; /* end class RBoundary */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
