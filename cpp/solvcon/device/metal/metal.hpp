#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

// forward declaration.
namespace MTL
{
class Device;
} /* end namespace MTL */

namespace solvcon
{

namespace device
{

class MetalManager
{

public:

    static MetalManager & instance();

    MetalManager(MetalManager const &) = delete;
    MetalManager(MetalManager &&) = delete;
    MetalManager & operator=(MetalManager const &) = delete;
    MetalManager & operator=(MetalManager &&) = delete;
    ~MetalManager() { shutdown(); }

    void startup();
    bool started() { return nullptr != m_device; }
    void shutdown();

private:

    MetalManager() { startup(); }

    MTL::Device * m_device = nullptr;

}; /* end class MetalManager */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
