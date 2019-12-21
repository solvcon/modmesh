#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>

namespace modmesh
{

/**
 * Spatial table basic information.  Any table-based data store for spatial
 * data should inherit this class template.
 */
template <size_t ND>
class SpaceBase
{

public:

    static constexpr const size_t NDIM = ND;

    using serial_type = uint32_t;
    using real_type = double;

}; /* end class SpaceBase */

/**
 * Base class template for structured grid.
 */
template <size_t ND>
class GridBase
  : public SpaceBase<ND>
{
}; /* end class GridBase */

/**
 * 1D grid.
 */
class Grid1d
  : public GridBase<1>
{
}; /* end class Grid1d */

class Grid2d
  : public GridBase<2>
{
}; /* end class Grid2d */

class Grid3d
  : public GridBase<3>
{
}; /* end class Grid3d */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
