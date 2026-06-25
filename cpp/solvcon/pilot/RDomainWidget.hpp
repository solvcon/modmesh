#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <rhi/qrhi.h>

#include <QImage>
#include <QMatrix4x4>
#include <QRhiWidget>
#include <QVector3D>

#include <memory>
#include <vector>

namespace solvcon
{

/**
 * @brief Interactive 2D/3D viewer for spatial domains and fields on
 * unstructured meshes, rendered with QRhi and controlled from Python.
 *
 * This is the QRhi reimplementation of the pilot 3D viewer. It is a
 * QRhiWidget: Qt owns the swapchain, color and depth buffers, and drives the
 * render loop through initialize()/render(). The widget holds a list of
 * RDrawable objects (the mesh wireframe and, later, fields and highlights)
 * and a fit-to-domain view-projection. It is built side by side with the
 * legacy Qt 3D prototype while the latter is being retired.
 */
class RDomainWidget
    : public QRhiWidget
{
    Q_OBJECT

public:

    explicit RDomainWidget(QWidget * parent = nullptr);
    ~RDomainWidget() override;

    /// Replace the rendered mesh with the wireframe of @p mesh.
    void updateMesh(std::shared_ptr<StaticMesh> const & mesh);

    /// Show or hide the mesh wireframe.
    void showMesh(bool show);

    std::shared_ptr<StaticMesh> mesh() const { return m_mesh; }

    /// Render the current frame offscreen and return it as a QImage. Thin
    /// wrapper over QRhiWidget::grabFramebuffer() for the Python control path.
    QImage grabImage();

protected:

    void initialize(QRhiCommandBuffer * cb) override;
    void render(QRhiCommandBuffer * cb) override;
    void releaseResources() override;

private:

    /// Build a view-projection that frames the domain bounding box into the
    /// given pixel viewport. A precursor to the dedicated scene framing.
    QMatrix4x4 computeViewProj(QSize pixel_size) const;

    QRhi * m_rhi = nullptr; ///< Tracked to detect device changes.
    QRhiRenderPassDescriptor * m_rpdesc = nullptr; ///< Tracked to detect target changes.
    int m_sample_count = 0; ///< Tracked to detect MSAA changes.

    std::vector<std::unique_ptr<RDrawable>> m_drawables;
    RDrawable * m_mesh_frame = nullptr; ///< Non-owning; lives in m_drawables.

    std::shared_ptr<StaticMesh> m_mesh;

    QVector3D m_bbox_lo;
    QVector3D m_bbox_hi;
    uint32_t m_ndim = 0;
    bool m_has_bbox = false;

}; /* end class RDomainWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
