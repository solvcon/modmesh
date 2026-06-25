#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RMaterial.hpp>

#include <rhi/qrhi.h>

#include <QImage>
#include <QMatrix4x4>
#include <QRhiWidget>

#include <memory>

namespace solvcon
{

/**
 * @brief Interactive 2D/3D viewer for spatial domains and fields on
 * unstructured meshes, rendered with QRhi and controlled from Python.
 *
 * This is the QRhi reimplementation of the pilot 3D viewer. It is a
 * QRhiWidget: Qt owns the swapchain, color and depth buffers, and drives the
 * render loop through initialize()/render(). The widget is built side by side
 * with the legacy Qt 3D prototype while the latter is being retired.
 */
class RDomainWidget
    : public QRhiWidget
{
    Q_OBJECT

public:

    explicit RDomainWidget(QWidget * parent = nullptr);
    ~RDomainWidget() override;

    /// Render the current frame offscreen and return it as a QImage. Thin
    /// wrapper over QRhiWidget::grabFramebuffer() for the Python control path.
    QImage grabImage();

protected:

    void initialize(QRhiCommandBuffer * cb) override;
    void render(QRhiCommandBuffer * cb) override;
    void releaseResources() override;

private:

    /// (Re)create the device-owned resources after an rhi or target change.
    void buildResources();

    QRhi * m_rhi = nullptr; ///< Tracked to detect device changes.

    std::unique_ptr<RMaterial> m_material;
    std::unique_ptr<QRhiBuffer> m_vbuf;
    std::unique_ptr<QRhiBuffer> m_ubuf;
    std::unique_ptr<QRhiShaderResourceBindings> m_srb;
    std::unique_ptr<QRhiGraphicsPipeline> m_pipeline;

    bool m_vbuf_uploaded = false;

}; /* end class RDomainWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
