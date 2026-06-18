#pragma once

/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/pilot/common_detail.hpp> // Must be the first include.

#include <modmesh/universe/universe.hpp>
#include <modmesh/pilot/R3DWidget.hpp>

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

namespace modmesh
{

class RVertices
    : public Qt3DCore::QEntity
{
    Q_OBJECT

public:

    RVertices(std::shared_ptr<WorldFp64> const & world, Qt3DCore::QNode * parent = nullptr);

private:

    Qt3DCore::QGeometry * m_geometry = nullptr;
    Qt3DRender::QGeometryRenderer * m_renderer = nullptr;
    Qt3DRender::QMaterial * m_material = nullptr;

}; /* end class RVertices */

class RLines
    : public Qt3DCore::QEntity
{
    Q_OBJECT

public:

    RLines(std::shared_ptr<WorldFp64> const & world, Qt3DCore::QNode * parent = nullptr);

    void update();

private:

    Qt3DCore::QGeometry * m_geometry = nullptr;
    Qt3DRender::QGeometryRenderer * m_renderer = nullptr;
    Qt3DRender::QMaterial * m_material = nullptr;

}; /* end class RLines */

class RColorField
    : public Qt3DCore::QEntity
{
    Q_OBJECT

public:

    RColorField(
        SimpleArray<float> const & vertices,
        SimpleArray<float> const & colors,
        SimpleArray<uint32_t> const & indices,
        Qt3DCore::QNode * parent = nullptr);

private:

    Qt3DCore::QGeometry * m_geometry = nullptr;
    Qt3DRender::QGeometryRenderer * m_renderer = nullptr;
    Qt3DRender::QMaterial * m_material = nullptr;

}; /* end class RColorField */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
