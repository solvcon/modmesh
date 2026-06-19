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

#include <Qt3DRender/QLayer>

#include <Qt3DExtras/QDiffuseSpecularMaterial>

namespace Qt3DExtras
{

class QConeMesh;

}

namespace solvcon
{

class RArrowHead
    : public Qt3DCore::QEntity
{

public:

    RArrowHead(QVector3D const & v0, QVector3D const & v1, QColor const & color, Qt3DCore::QNode * parent = nullptr);

    void setColor(QColor const & color);
    float length() const;
    void setLength(float v);
    float bottomRadius() const;
    void setBottomRadius(float v);
    void setBottomRadiusRatio(float v);

private:

    Qt3DExtras::QConeMesh * m_renderer = nullptr;
    Qt3DExtras::QDiffuseSpecularMaterial * m_material = nullptr;

}; /* end class RArrowHead */

class RLine
    : public Qt3DCore::QEntity
{

public:

    RLine(QVector3D const & v0, QVector3D const & v1, QColor const & color, Qt3DCore::QNode * parent = nullptr);

    void addArrowHead(float erate, float wrate);

    QColor color() const;
    void setColor(QColor const & color);

private:

    Qt3DCore::QGeometry * m_geometry = nullptr;
    Qt3DRender::QGeometryRenderer * m_renderer = nullptr;
    Qt3DExtras::QDiffuseSpecularMaterial * m_material = nullptr;
    RArrowHead * m_arrow_head = nullptr;
    QVector3D m_v0;
    QVector3D m_v1;

}; /* end class RLine */

class RAxisMark
    : public Qt3DCore::QEntity
{

public:

    RAxisMark(Qt3DCore::QNode * parent = nullptr, Qt3DRender::QLayer * layer = nullptr);

private:

    RLine * m_xmark = nullptr;
    RLine * m_ymark = nullptr;
    RLine * m_zmark = nullptr;

    QEntity * m_xtext = nullptr;
    QEntity * m_ytext = nullptr;
    QEntity * m_ztext = nullptr;

}; /* end class RAxisMark */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
