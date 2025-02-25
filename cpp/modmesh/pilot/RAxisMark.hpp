#pragma once

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

#include <modmesh/pilot/common_detail.hpp> // Must be the first include.

#include <modmesh/modmesh.hpp>

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

namespace Qt3DExtras
{

class QConeMesh;

}

namespace modmesh
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

    RAxisMark(Qt3DCore::QNode * parent = nullptr);

private:

    RLine * m_xmark = nullptr;
    RLine * m_ymark = nullptr;
    RLine * m_zmark = nullptr;

    QEntity * m_xtext = nullptr;
    QEntity * m_ytext = nullptr;
    QEntity * m_ztext = nullptr;

}; /* end class RAxisMark */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
