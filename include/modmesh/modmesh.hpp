#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

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
class GridD1
  : public GridBase<1>
{
}; /* end class GridD1 */

class GridD2
  : public GridBase<2>
{
}; /* end class GridD2 */

class GridD3
  : public GridBase<3>
{
}; /* end class GridD3 */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
