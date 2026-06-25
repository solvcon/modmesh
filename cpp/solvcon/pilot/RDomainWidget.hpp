#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RDomainScene.hpp>
#include <solvcon/pilot/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <rhi/qrhi.h>

#include <QImage>
#include <QPoint>
#include <QRhiWidget>
#include <QVector3D>

#include <memory>
#include <string>

namespace solvcon
{

/**
 * @brief Interactive 2D/3D viewer for spatial domains and fields on
 * unstructured meshes, rendered with QRhi and controlled from Python.
 *
 * This is the QRhi reimplementation of the pilot 3D viewer. It is a
 * QRhiWidget: Qt owns the swapchain, color and depth buffers, and drives the
 * render loop through initialize()/render(). The widget hosts an RDomainScene
 * (the drawables, the domain bounding box, and the framing camera) and drives
 * it. It is built side by side with the legacy Qt 3D prototype while the
 * latter is being retired.
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

    /// Replace the colored field: per-vertex-colored triangles from a vertex
    /// table (nvert, 3), a matching color table (nvert, 3), and a triangle
    /// index table (ntri, 3). Swappable at runtime.
    void updateColorField(
        SimpleArray<float> const & vertices,
        SimpleArray<float> const & colors,
        SimpleArray<uint32_t> const & indices);

    /// Show or hide the highlight ribbon for boundary set @p ibc.
    void showBoundary(int ibc, bool show);

    /// Frame the camera so the whole domain is in view.
    void fitCameraToScene();

    /// Select the camera mode: "pan" (2D pan/zoom) or "fps" (3D fly-through).
    void setCameraMode(std::string const & name);
    std::string cameraMode() const;

    // Programmatic camera pose, so Python navigates as well as the mouse.
    QVector3D cameraPosition() const;
    void setCameraPosition(QVector3D const & position);
    QVector3D cameraTarget() const;
    void setCameraTarget(QVector3D const & target);
    QVector3D cameraUp() const;
    void setCameraUp(QVector3D const & up);

    // Mode-aware interaction primitives (what the mouse and wheel drive).
    void rotateCamera(float dx, float dy);
    void panCamera(float dx, float dy);
    void zoomCamera(float steps);

    std::shared_ptr<StaticMesh> mesh() const { return m_mesh; }

    /// Render the current frame offscreen and return it as a QImage. Thin
    /// wrapper over QRhiWidget::grabFramebuffer() for the Python control path.
    QImage grabImage();

protected:

    void initialize(QRhiCommandBuffer * cb) override;
    void render(QRhiCommandBuffer * cb) override;
    void releaseResources() override;

    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;
    void wheelEvent(QWheelEvent * event) override;
    void keyPressEvent(QKeyEvent * event) override;

private:

    float viewportAspect() const;

    QRhi * m_rhi = nullptr; ///< Tracked to detect device changes.
    QRhiRenderPassDescriptor * m_rpdesc = nullptr; ///< Tracked to detect target changes.
    int m_sample_count = 0; ///< Tracked to detect MSAA changes.

    RDomainScene m_scene;
    RDrawable * m_mesh_frame = nullptr; ///< Non-owning; lives in the scene.
    RDrawable * m_field = nullptr; ///< Non-owning; lives in the scene.

    std::shared_ptr<StaticMesh> m_mesh;

    QPoint m_last_mouse_pos; ///< Last cursor position during a drag.
    bool m_panning = false; ///< A non-left-button drag pans in both modes.

}; /* end class RDomainWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
