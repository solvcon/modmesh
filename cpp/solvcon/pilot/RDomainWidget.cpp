/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RDomainWidget.hpp> // Must be the first include.

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
    // Drop the previous mesh wireframe and replace it.
    if (nullptr != m_mesh_frame)
    {
        auto it = std::find_if(
            m_drawables.begin(),
            m_drawables.end(),
            [this](std::unique_ptr<RDrawable> const & d)
            { return d.get() == m_mesh_frame; });
        if (it != m_drawables.end())
        {
            m_drawables.erase(it);
        }
        m_mesh_frame = nullptr;
    }

    m_mesh = mesh;

    auto frame = std::make_unique<RMeshFrame>(mesh);
    m_mesh_frame = frame.get();
    m_drawables.push_back(std::move(frame));

    // Recompute the domain bounding box for framing.
    StaticMesh const & mh = *mesh;
    m_ndim = mh.ndim();
    QVector3D lo(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    QVector3D hi(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        float const x = static_cast<float>(mh.ndcrd(ind, 0));
        float const y = static_cast<float>(mh.ndcrd(ind, 1));
        float const z = (3 == m_ndim) ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f;
        lo = QVector3D(std::min(lo.x(), x), std::min(lo.y(), y), std::min(lo.z(), z));
        hi = QVector3D(std::max(hi.x(), x), std::max(hi.y(), y), std::max(hi.z(), z));
    }
    m_bbox_lo = lo;
    m_bbox_hi = hi;
    m_has_bbox = (mh.nnode() > 0);

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

QMatrix4x4 RDomainWidget::computeViewProj(QSize pixel_size) const
{
    QMatrix4x4 clip = m_rhi ? m_rhi->clipSpaceCorrMatrix() : QMatrix4x4();
    if (!m_has_bbox || pixel_size.height() <= 0 || pixel_size.width() <= 0)
    {
        return clip;
    }

    QVector3D const center = (m_bbox_lo + m_bbox_hi) * 0.5f;
    QVector3D const extent = m_bbox_hi - m_bbox_lo;
    float radius = extent.length() * 0.5f;
    if (radius <= 0.0f)
    {
        radius = 1.0f;
    }

    // A bounding sphere of this radius fits the domain from any view
    // direction, so framing is just an orthographic box around it.
    float const aspect = static_cast<float>(pixel_size.width()) / static_cast<float>(pixel_size.height());
    float const margin = radius * 1.1f;
    float half_w = margin;
    float half_h = margin;
    if (aspect >= 1.0f)
    {
        half_w = margin * aspect;
    }
    else
    {
        half_h = margin / aspect;
    }

    // 2D domains are viewed head-on; 3D domains from a fixed oblique angle so
    // depth reads until the interactive camera lands.
    QVector3D const dir = (3 == m_ndim)
                              ? QVector3D(0.6f, 0.5f, 1.0f).normalized()
                              : QVector3D(0.0f, 0.0f, 1.0f);
    QVector3D const eye = center + dir * (2.0f * radius);

    QMatrix4x4 view;
    view.lookAt(eye, center, QVector3D(0.0f, 1.0f, 0.0f));

    QMatrix4x4 proj;
    proj.ortho(-half_w, half_w, -half_h, half_h, 0.01f * radius, 5.0f * radius);

    return clip * proj * view;
}

void RDomainWidget::initialize(QRhiCommandBuffer *)
{
    QRhiRenderPassDescriptor * const rpdesc = renderTarget()->renderPassDescriptor();
    if (m_rhi != rhi() || m_rpdesc != rpdesc || m_sample_count != sampleCount())
    {
        // The graphics device, render target, or sample count changed; drop
        // every device resource so the drawables rebuild against the new one
        // (the pipelines are tied to the render-pass descriptor).
        for (std::unique_ptr<RDrawable> const & drawable : m_drawables)
        {
            drawable->release();
        }
        m_rhi = rhi();
        m_rpdesc = rpdesc;
        m_sample_count = sampleCount();
    }
}

void RDomainWidget::render(QRhiCommandBuffer * cb)
{
    QRhiResourceUpdateBatch * batch = m_rhi->nextResourceUpdateBatch();

    QSize const pixel_size = renderTarget()->pixelSize();
    QMatrix4x4 const view_proj = computeViewProj(pixel_size);

    for (std::unique_ptr<RDrawable> const & drawable : m_drawables)
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
    for (std::unique_ptr<RDrawable> const & drawable : m_drawables)
    {
        drawable->draw(cb);
    }
    cb->endPass();
}

void RDomainWidget::releaseResources()
{
    for (std::unique_ptr<RDrawable> const & drawable : m_drawables)
    {
        drawable->release();
    }
    m_rhi = nullptr;
    m_rpdesc = nullptr;
    m_sample_count = 0;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
