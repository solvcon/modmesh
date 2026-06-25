/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RMeshFrame.hpp> // Must be the first include.

namespace solvcon
{

RMeshFrame::RMeshFrame(std::shared_ptr<StaticMesh> const & mesh)
{
    StaticMesh const & mh = *mesh;
    uint32_t const nnode = mh.nnode();
    uint32_t const ndim = mh.ndim();

    m_positions.reserve(static_cast<size_t>(nnode) * 3);
    for (uint32_t ind = 0; ind < nnode; ++ind)
    {
        m_positions.push_back(static_cast<float>(mh.ndcrd(ind, 0)));
        m_positions.push_back(static_cast<float>(mh.ndcrd(ind, 1)));
        m_positions.push_back((3 == ndim) ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f);
    }

    uint32_t const nedge = mh.nedge();
    m_indices.reserve(static_cast<size_t>(nedge) * 2);
    for (uint32_t ie = 0; ie < nedge; ++ie)
    {
        m_indices.push_back(static_cast<uint32_t>(mh.ednds(ie, 0)));
        m_indices.push_back(static_cast<uint32_t>(mh.ednds(ie, 1)));
    }

    // A light hairline color over the dark background.
    setColor(QVector4D(0.82f, 0.86f, 0.92f, 1.0f));
}

QRhiVertexInputLayout RMeshFrame::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    layout.setBindings({{3 * sizeof(float)}});
    layout.setAttributes({{0, 0, QRhiVertexInputAttribute::Float3, 0}});
    return layout;
}

void RMeshFrame::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
{
    if (0 == m_positions.size() || 0 == m_indices.size())
    {
        return;
    }

    quint32 const vbytes = static_cast<quint32>(m_positions.size() * sizeof(float));
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, vbytes));
    m_vbuf->create();
    batch->uploadStaticBuffer(m_vbuf.get(), m_positions.data());
    m_vertex_count = static_cast<quint32>(m_positions.size() / 3);

    quint32 const ibytes = static_cast<quint32>(m_indices.size() * sizeof(uint32_t));
    m_ibuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::IndexBuffer, ibytes));
    m_ibuf->create();
    batch->uploadStaticBuffer(m_ibuf.get(), m_indices.data());
    m_index_count = static_cast<quint32>(m_indices.size());
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
