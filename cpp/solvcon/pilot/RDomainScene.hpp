#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RDrawable.hpp>

#include <rhi/qrhi.h>

#include <QMatrix4x4>
#include <QSize>
#include <QVector3D>

#include <functional>
#include <memory>
#include <vector>

namespace solvcon
{

/**
 * @brief The renderable scene: the drawables, the domain bounding box, and
 * the framing camera with a per-dimension projection.
 *
 * The scene owns the list of RDrawable objects and the domain bounding box,
 * picks an orthographic projection for 2D domains and a perspective
 * projection for 3D ones, and frames the camera onto the box. RDomainWidget
 * holds one scene and drives it; the scene itself is free of Qt widget and
 * device-loop concerns.
 */
class RDomainScene
{

public:

    RDomainScene() = default;

    RDomainScene(RDomainScene const &) = delete;
    RDomainScene & operator=(RDomainScene const &) = delete;

    /// Add a drawable; ownership transfers to the scene.
    void addDrawable(std::unique_ptr<RDrawable> drawable);

    /// Remove a specific drawable (no-op for nullptr or a foreign pointer).
    void removeDrawable(RDrawable const * drawable);

    /// Remove every drawable the predicate accepts.
    void removeDrawableIf(std::function<bool(RDrawable const *)> const & pred);

    std::vector<std::unique_ptr<RDrawable>> & drawables() { return m_drawables; }

    /// Release the device resources of every drawable.
    void releaseAll();

    /// Forget the bounding box so the next extendBoundingBox sets it afresh.
    void resetBoundingBox();

    /// Grow the bounding box to include the point range [lo, hi].
    void extendBoundingBox(QVector3D const & lo, QVector3D const & hi);

    bool hasBoundingBox() const { return m_has_bbox; }

    void setDimension(uint32_t ndim) { m_ndim = ndim; }
    uint32_t dimension() const { return m_ndim; }

    /// Frame the camera so the whole bounding box is in view.
    void fitCameraToScene();

    /// The model-view-projection for the framed scene at the given viewport.
    /// 2D domains use an orthographic projection, 3D domains a perspective
    /// one. @p rhi supplies the backend clip-space correction.
    QMatrix4x4 viewProjection(QSize pixel_size, QRhi * rhi) const;

private:

    float boundingRadius() const;

    std::vector<std::unique_ptr<RDrawable>> m_drawables;

    QVector3D m_bbox_lo;
    QVector3D m_bbox_hi;
    bool m_has_bbox = false;
    uint32_t m_ndim = 0;

    QVector3D m_eye{0.0f, 0.0f, 1.0f};
    QVector3D m_center{0.0f, 0.0f, 0.0f};
    QVector3D m_up{0.0f, 1.0f, 0.0f};

}; /* end class RDomainScene */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
