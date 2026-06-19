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

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
