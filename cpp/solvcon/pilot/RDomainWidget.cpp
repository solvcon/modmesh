/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RDomainWidget.hpp> // Must be the first include.

namespace solvcon
{

namespace
{

// std140 layout of the FlatColor uniform block: mat4 mvp + vec4 color.
constexpr int UBUF_SIZE = 64 + 16;

// A single triangle in normalized device coordinates. Step 1 draws this lone
// primitive to prove the render foundation and the Python control spine.
constexpr float TRIANGLE_VERTICES[] = {
    // clang-format off
     0.0f,  0.5f, 0.0f,
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
    // clang-format on
};

} /* end namespace */

RDomainWidget::RDomainWidget(QWidget * parent)
    : QRhiWidget(parent)
{
}

RDomainWidget::~RDomainWidget() = default;

QImage RDomainWidget::grabImage()
{
    return grabFramebuffer();
}

void RDomainWidget::buildResources()
{
    m_vbuf.reset(m_rhi->newBuffer(
        QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, sizeof(TRIANGLE_VERTICES)));
    m_vbuf->create();
    m_vbuf_uploaded = false;

    m_ubuf.reset(m_rhi->newBuffer(
        QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, UBUF_SIZE));
    m_ubuf->create();

    m_srb.reset(m_rhi->newShaderResourceBindings());
    m_srb->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            m_ubuf.get()),
    });
    m_srb->create();

    m_material = std::make_unique<RMaterial>(RMaterial::Kind::FlatColor);

    QRhiVertexInputLayout input_layout;
    input_layout.setBindings({{3 * sizeof(float)}});
    input_layout.setAttributes({{0, 0, QRhiVertexInputAttribute::Float3, 0}});

    m_pipeline.reset(m_material->buildPipeline(
        m_rhi,
        m_srb.get(),
        renderTarget()->renderPassDescriptor(),
        input_layout,
        QRhiGraphicsPipeline::Triangles,
        sampleCount()));
    if (!m_pipeline)
    {
        qWarning("RDomainWidget: failed to build the graphics pipeline");
    }
}

void RDomainWidget::initialize(QRhiCommandBuffer *)
{
    if (m_rhi != rhi())
    {
        // The rhi (graphics device) was created or replaced; drop everything
        // tied to the old device and rebuild against the new one.
        m_pipeline.reset();
        m_srb.reset();
        m_ubuf.reset();
        m_vbuf.reset();
        m_material.reset();
        m_rhi = rhi();
    }

    if (!m_pipeline)
    {
        buildResources();
    }
}

void RDomainWidget::render(QRhiCommandBuffer * cb)
{
    if (!m_pipeline)
    {
        return;
    }

    QRhiResourceUpdateBatch * batch = m_rhi->nextResourceUpdateBatch();

    if (!m_vbuf_uploaded)
    {
        batch->uploadStaticBuffer(m_vbuf.get(), TRIANGLE_VERTICES);
        m_vbuf_uploaded = true;
    }

    QMatrix4x4 mvp = m_rhi->clipSpaceCorrMatrix();
    batch->updateDynamicBuffer(m_ubuf.get(), 0, 64, mvp.constData());
    float const color[4] = {0.2f, 0.6f, 0.9f, 1.0f};
    batch->updateDynamicBuffer(m_ubuf.get(), 64, 16, color);

    QColor const clear_color = QColor::fromRgbF(0.12f, 0.12f, 0.14f, 1.0f);
    QRhiDepthStencilClearValue const ds_clear(1.0f, 0);

    QSize const pixel_size = renderTarget()->pixelSize();
    cb->beginPass(renderTarget(), clear_color, ds_clear, batch);
    cb->setGraphicsPipeline(m_pipeline.get());
    cb->setViewport(QRhiViewport(
        0, 0, float(pixel_size.width()), float(pixel_size.height())));
    cb->setShaderResources();
    QRhiCommandBuffer::VertexInput const vbuf_binding(m_vbuf.get(), 0);
    cb->setVertexInput(0, 1, &vbuf_binding);
    cb->draw(3);
    cb->endPass();
}

void RDomainWidget::releaseResources()
{
    m_pipeline.reset();
    m_srb.reset();
    m_ubuf.reset();
    m_vbuf.reset();
    m_material.reset();
    m_rhi = nullptr;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
