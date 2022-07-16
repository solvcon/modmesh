#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/spacetime/ElementBase_decl.hpp>
#include <modmesh/spacetime/Field_decl.hpp>
#include <modmesh/spacetime/Grid_decl.hpp>

namespace spacetime
{

template <class ET>
inline Grid const & ElementBase<ET>::grid() const { return m_field->grid(); }

template <class ET>
inline size_t ElementBase<ET>::xindex() const { return m_xptr - grid().xptr(); }

} /* end namespace spacetime */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
