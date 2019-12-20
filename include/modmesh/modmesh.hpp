/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

namespace modmesh
{

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

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
