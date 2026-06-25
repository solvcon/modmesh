/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RDomainWidget.hpp> // Must be the first include.

#include <solvcon/pilot/RField.hpp>
#include <solvcon/pilot/RMeshBoundary.hpp>
#include <solvcon/pilot/RMeshFrame.hpp>

#include <algorithm>
#include <limits>

namespace solvcon
{

RDomainWidget::RDomainWidget(QWidget * parent)
    : QRhiWidget(parent)
{
}

RDomainWidget::~RDomainWidget() = default;

QImage RDomainWidget::grabImage()
{
    return grabFramebuffer();
}

void RDomainWidget::updateMesh(std::shared_ptr<StaticMesh> const & mesh)
{
    // Drop the previous mesh wireframe and replace it; a new mesh redefines
    // the framing, so the bounding box is recomputed from scratch.
    m_scene.removeDrawable(m_mesh_frame);
    m_mesh_frame = nullptr;

    m_mesh = mesh;

    auto frame = std::make_unique<RMeshFrame>(mesh);
    m_mesh_frame = frame.get();
    m_scene.addDrawable(std::move(frame));

    StaticMesh const & mh = *mesh;
    m_scene.setDimension(mh.ndim());
    QVector3D lo(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    QVector3D hi(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());
    bool const is_3d = (3 == mh.ndim());
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        float const x = static_cast<float>(mh.ndcrd(ind, 0));
        float const y = static_cast<float>(mh.ndcrd(ind, 1));
        float const z = is_3d ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f;
        lo = QVector3D(std::min(lo.x(), x), std::min(lo.y(), y), std::min(lo.z(), z));
        hi = QVector3D(std::max(hi.x(), x), std::max(hi.y(), y), std::max(hi.z(), z));
    }
    m_scene.resetBoundingBox();
    if (mh.nnode() > 0)
    {
        m_scene.extendBoundingBox(lo, hi);
    }
    // A field set earlier still draws, so keep it inside the framed box.
    if (nullptr != m_field)
    {
        auto * field = static_cast<RField *>(m_field);
        m_scene.extendBoundingBox(field->bboxLo(), field->bboxHi());
    }

    m_scene.fitCameraToScene();
    update();
}

void RDomainWidget::showMesh(bool show)
{
    if (nullptr != m_mesh_frame)
    {
        m_mesh_frame->setVisible(show);
        update();
    }
}

void RDomainWidget::updateColorField(
    SimpleArray<float> const & vertices,
    SimpleArray<float> const & colors,
    SimpleArray<uint32_t> const & indices)
{
    // Drop the previous field and replace it; the field is swappable.
    m_scene.removeDrawable(m_field);
    m_field = nullptr;

    auto field = std::make_unique<RField>(vertices, colors, indices);
    if (field->hasGeometry())
    {
        QVector3D const lo = field->bboxLo();
        QVector3D const hi = field->bboxHi();
        // With no mesh to set the dimensionality, infer it: a field with no
        // depth extent is viewed head-on like a 2D domain.
        if (!m_scene.hasBoundingBox())
        {
            float const span = (hi - lo).length();
            m_scene.setDimension(((hi.z() - lo.z()) > 1.0e-6f * span) ? 3 : 2);
        }
        m_scene.extendBoundingBox(lo, hi);
        m_field = field.get();
        m_scene.addDrawable(std::move(field));
        m_scene.fitCameraToScene();
    }

    update();
}

void RDomainWidget::showBoundary(int ibc, bool show)
{
    // Remove an existing highlight for this set so a re-show stays single and
    // a hide leaves none behind.
    m_scene.removeDrawableIf(
        [ibc](RDrawable const * d)
        {
            auto const * boundary = dynamic_cast<RMeshBoundary const *>(d);
            return nullptr != boundary && boundary->ibc() == ibc;
        });

    if (show && nullptr != m_mesh)
    {
        auto boundary = std::make_unique<RMeshBoundary>(m_mesh, ibc);
        if (boundary->hasGeometry())
        {
            m_scene.addDrawable(std::move(boundary));
        }
    }

    update();
}

void RDomainWidget::fitCameraToScene()
{
    m_scene.fitCameraToScene();
    update();
}

void RDomainWidget::initialize(QRhiCommandBuffer *)
{
    QRhiRenderPassDescriptor * const rpdesc = renderTarget()->renderPassDescriptor();
    if (m_rhi != rhi() || m_rpdesc != rpdesc || m_sample_count != sampleCount())
    {
        // The graphics device, render target, or sample count changed; drop
        // every device resource so the drawables rebuild against the new one
        // (the pipelines are tied to the render-pass descriptor).
        m_scene.releaseAll();
        m_rhi = rhi();
        m_rpdesc = rpdesc;
        m_sample_count = sampleCount();
    }
}

void RDomainWidget::render(QRhiCommandBuffer * cb)
{
    QRhiResourceUpdateBatch * batch = m_rhi->nextResourceUpdateBatch();

    QSize const pixel_size = renderTarget()->pixelSize();
    QMatrix4x4 const view_proj = m_scene.viewProjection(pixel_size, m_rhi);

    for (std::unique_ptr<RDrawable> const & drawable : m_scene.drawables())
    {
        drawable->prepare(
            m_rhi, renderTarget()->renderPassDescriptor(), sampleCount(), batch);
        drawable->updateUniform(batch, view_proj);
    }

    QColor const clear_color = QColor::fromRgbF(0.12f, 0.12f, 0.14f, 1.0f);
    QRhiDepthStencilClearValue const ds_clear(1.0f, 0);

    cb->beginPass(renderTarget(), clear_color, ds_clear, batch);
    cb->setViewport(QRhiViewport(
        0, 0, float(pixel_size.width()), float(pixel_size.height())));
    for (std::unique_ptr<RDrawable> const & drawable : m_scene.drawables())
    {
        drawable->draw(cb);
    }
    cb->endPass();
}

void RDomainWidget::releaseResources()
{
    m_scene.releaseAll();
    m_rhi = nullptr;
    m_rpdesc = nullptr;
    m_sample_count = 0;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
