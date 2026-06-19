/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.
#include <solvcon/pilot/RAxisMark.hpp>

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

#include <Qt3DExtras/QConeMesh>
#include <Qt3DExtras/QDiffuseSpecularMaterial>
#include <Qt3DExtras/QExtrudedTextMesh>

namespace solvcon
{

RArrowHead::RArrowHead(QVector3D const & v0, QVector3D const & v1, QColor const & color, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_renderer(new Qt3DExtras::QConeMesh())
    , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
{
    QVector3D const vec = v1 - v0;
    m_renderer->setLength(vec.length());
    addComponent(m_renderer);

    auto * transform = new Qt3DCore::QTransform();
    transform->setRotation(QQuaternion::rotationTo(QVector3D(0.0f, 1.0f, 0.0f), vec));
    transform->setScale(1.0f);
    transform->setTranslation(v0 + vec / 2);
    addComponent(transform);

    m_material->setAmbient(color);
    addComponent(m_material);
}

void RArrowHead::setColor(QColor const & color) { m_material->setAmbient(color); }

float RArrowHead::length() const { return m_renderer->length(); }

void RArrowHead::setLength(float v) { m_renderer->setLength(v); }

float RArrowHead::bottomRadius() const { return m_renderer->bottomRadius(); }

void RArrowHead::setBottomRadius(float v) { m_renderer->setBottomRadius(v); }

void RArrowHead::setBottomRadiusRatio(float v) { setBottomRadius(length() * v); }

QColor RLine::color() const { return m_material->ambient(); }

void RLine::setColor(QColor const & color)
{
    m_material->setAmbient(color);
    if (m_arrow_head)
    {
        m_arrow_head->setColor(color);
    }
}

void RLine::addArrowHead(float erate, float wrate)
{
    if (!m_arrow_head)
    {
        QVector3D v2 = (m_v1 - m_v0) * erate;
        v2 = m_v1 - v2;
        m_arrow_head = new RArrowHead(v2, m_v1, color(), this);
        m_arrow_head->setBottomRadiusRatio(wrate);
    }
}

RLine::RLine(QVector3D const & v0, QVector3D const & v1, QColor const & color, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_geometry(new Qt3DCore::QGeometry(this))
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
    , m_v0(v0)
    , m_v1(v1)
{
    {
        auto * buf = new Qt3DCore::QBuffer(m_geometry);
        {
            QByteArray barray;
            barray.resize(2 * 3 * sizeof(float));
            float * ptr = reinterpret_cast<float *>(barray.data());
            ptr[0] = v0.x();
            ptr[1] = v0.y();
            ptr[2] = v0.z();
            ptr[3] = v1.x();
            ptr[4] = v1.y();
            ptr[5] = v1.z();
            buf->setData(barray);
        }

        {
            auto * vertices = new Qt3DCore::QAttribute(m_geometry);
            vertices->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
            vertices->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
            vertices->setVertexBaseType(Qt3DCore::QAttribute::Float);
            vertices->setVertexSize(3);
            vertices->setBuffer(buf);
            vertices->setByteStride(3 * sizeof(float));
            vertices->setCount(2);
            m_geometry->addAttribute(vertices);
        }
    }

    {
        auto * buf = new Qt3DCore::QBuffer(m_geometry);
        {
            QByteArray barray;
            barray.resize(2 * sizeof(uint32_t));
            auto * indices = reinterpret_cast<uint32_t *>(barray.data());
            indices[0] = 0;
            indices[1] = 1;
            buf->setData(barray);
        }

        {
            auto * indices = new Qt3DCore::QAttribute(m_geometry);
            indices->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
            indices->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
            indices->setBuffer(buf);
            indices->setCount(2);
            m_geometry->addAttribute(indices);
        }
    }

    m_renderer->setGeometry(m_geometry);
    m_renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
    addComponent(m_renderer);
    addComponent(m_material);
    m_material->setAmbient(color);
}

namespace detail
{

static Qt3DCore::QEntity * drawText(std::string const & text, QVector3D loc, float scale, QColor color, Qt3DCore::QEntity * parent = nullptr)
{
    auto * entity = new Qt3DCore::QEntity(parent);

    auto * transform = new Qt3DCore::QTransform(entity);
    transform->setTranslation(loc);
    transform->setScale(scale);
    entity->addComponent(transform);

    auto * mesh = new Qt3DExtras::QExtrudedTextMesh(entity);
    mesh->setDepth(0.0f);
    mesh->setFont(QFont("Courier New", 10, -1, false));
    mesh->setText(text.c_str());
    entity->addComponent(mesh);

    auto * material = new Qt3DExtras::QDiffuseSpecularMaterial(entity);
    material->setAmbient(color);
    entity->addComponent(material);

    return entity;
}

} // end namespace detail

RAxisMark::RAxisMark(Qt3DCore::QNode * parent, Qt3DRender::QLayer * layer)
    : Qt3DCore::QEntity(parent)
    , m_xmark(new RLine(QVector3D(0.0f, 0.0f, 0.0f), QVector3D(1.0f, 0.0f, 0.0f), Qt::red, this))
    , m_ymark(new RLine(QVector3D(0.0f, 0.0f, 0.0f), QVector3D(0.0f, 1.0f, 0.0f), Qt::green, this))
    , m_zmark(new RLine(QVector3D(0.0f, 0.0f, 0.0f), QVector3D(0.0f, 0.0f, 1.0f), Qt::blue, this))
    , m_xtext(detail::drawText("X", QVector3D{1.1f, 0.0f, 0.0f}, 0.2f, Qt::red, this))
    , m_ytext(detail::drawText("Y", QVector3D{0.0f, 1.1f, 0.0f}, 0.2f, Qt::green, this))
    , m_ztext(detail::drawText("Z", QVector3D{0.0f, 0.0f, 1.1f}, 0.2f, Qt::blue, this))
{
    m_xmark->addArrowHead(0.2f, 0.4f);
    m_ymark->addArrowHead(0.2f, 0.4f);
    m_zmark->addArrowHead(0.2f, 0.4f);

    // Tag the mark (recursively) so the gizmo framegraph branch can pick it out.
    if (nullptr != layer)
    {
        addComponent(layer);
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
