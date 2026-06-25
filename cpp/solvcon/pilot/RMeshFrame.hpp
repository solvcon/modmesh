#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <memory>

namespace solvcon
{

/**
 * @brief The unstructured-mesh domain rendered as a wireframe.
 *
 * Edges come from the mesh edge list (StaticMesh::ednds); the node
 * coordinates feed a line-topology vertex buffer. Works for both 2D meshes
 * (drawn in the z = 0 plane) and 3D meshes.
 */
class RMeshFrame
    : public RDrawable
{

public:

    explicit RMeshFrame(std::shared_ptr<StaticMesh> const & mesh);

protected:

    RMaterial::Kind materialKind() const override { return RMaterial::Kind::FlatColor; }

    QRhiGraphicsPipeline::Topology topology() const override { return QRhiGraphicsPipeline::Lines; }

    QRhiVertexInputLayout vertexInputLayout() const override;

    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;

private:

    // CPU-side geometry captured at construction; the rhi is not available
    // until prepare() runs, so the buffers are uploaded then.
    SimpleCollector<float> m_positions; ///< nnode * 3 (x, y, z).
    SimpleCollector<uint32_t> m_indices; ///< nedge * 2 node indices.

}; /* end class RMeshFrame */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
