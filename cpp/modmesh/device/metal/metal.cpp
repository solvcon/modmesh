/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include <Metal/Metal.hpp>
#pragma GCC diagnostic pop

#include <modmesh/device/metal/metal.hpp>

namespace modmesh
{

namespace device
{

MetalManager & MetalManager::instance()
{
    static MetalManager o;
    return o;
}

void MetalManager::startup()
{
    if (nullptr == m_device)
    {
        m_device = MTL::CreateSystemDefaultDevice();
    }
}

void MetalManager::shutdown()
{
    if (nullptr != m_device)
    {
        m_device->release();
        m_device = nullptr;
    }
}

} /* end namespace device */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
