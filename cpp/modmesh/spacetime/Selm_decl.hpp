#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/spacetime/base_spacetime.hpp>
#include <modmesh/spacetime/ElementBase_decl.hpp>
#include <modmesh/spacetime/Grid_decl.hpp>
#include <modmesh/spacetime/Field_decl.hpp>

namespace spacetime
{

/**
 * A solution element.
 */
class Selm
    : public ElementBase<Selm>
{

public:

    Selm(Field * field, size_t index, bool odd_plane)
        : base_type(field, field->grid().xptr_selm(index, odd_plane, Grid::SelmPK()))
    {
    }

    class const_ctor_passkey
    {
        const_ctor_passkey() = default;
        friend Field;
    };

    Selm(Field const * field, size_t index, bool odd_plane, const_ctor_passkey)
        // The only intention of this workaround is to let const Field to
        // create const object derived from Selm. Do not abuse.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        : Selm(const_cast<Field *>(field), index, odd_plane)
    {
    }

    int_type index() const
    {
        static_assert(0 == (Grid::BOUND_COUNT % 2), "only work with even BOUND_COUNT");
        return (static_cast<int_type>(xindex()) >> 1) - 1;
    }

    /**
     * Return true for even plane, false for odd plane (temporal).
     */
    bool on_even_plane() const { return !on_odd_plane(); }
    bool on_odd_plane() const { return static_cast<bool>((xindex() - Grid::BOUND_COUNT) & 1); }

    value_type dxneg() const { return x() - xneg(); }
    value_type dxpos() const { return xpos() - x(); }
    value_type xctr() const { return (xneg() + xpos()) / 2; }

    void move_at(int_type offset);

    value_type const & so0(size_t iv) const { return field().so0(xindex(), iv); }
    value_type & so0(size_t iv) { return field().so0(xindex(), iv); }

    value_type const & so1(size_t iv) const { return field().so1(xindex(), iv); }
    value_type & so1(size_t iv) { return field().so1(xindex(), iv); }

    value_type const & cfl() const { return field().cfl(xindex()); }
    value_type & cfl() { return field().cfl(xindex()); }

    value_type xn(size_t iv) const { return field().kernel().calc_xn(*this, iv); }
    value_type xp(size_t iv) const { return field().kernel().calc_xp(*this, iv); }
    value_type tn(size_t iv) const { return field().kernel().calc_tn(*this, iv); }
    value_type tp(size_t iv) const { return field().kernel().calc_tp(*this, iv); }
    value_type so0p(size_t iv) const { return field().kernel().calc_so0p(*this, iv); }
    void update_cfl() { return field().kernel().update_cfl(*this); }

}; /* end class Selm */

inline void Kernel::reset()
{
    m_xn_calc = [](Selm const &, size_t)
    { return 0.0; };
    m_xp_calc = [](Selm const &, size_t)
    { return 0.0; };
    m_tn_calc = [](Selm const &, size_t)
    { return 0.0; };
    m_tp_calc = [](Selm const &, size_t)
    { return 0.0; };
    m_so0p_calc = [](Selm const & se, size_t iv)
    { return se.so0(iv); };
    m_cfl_updater = [](Selm & se)
    { se.cfl() = 0.0; };
}

} /* end namespace spacetime */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
