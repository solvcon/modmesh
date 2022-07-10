#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/spacetime/system.hpp>
#include <modmesh/spacetime/type.hpp>
#include <modmesh/spacetime/ElementBase_decl.hpp>
#include <modmesh/spacetime/Grid_decl.hpp>
#include <modmesh/spacetime/SolverBase_decl.hpp>
#include <modmesh/spacetime/Selm_decl.hpp>

namespace spacetime
{

/* FIXME: The naming of Celm and CelmBase is confusing: Celm is the base class
 * of CelmBase. */

/**
 * A compound conservation celm.
 */
class Celm
    : public ElementBase<Celm>
{

public:

    using selm_type = Selm;

    Celm(Field * field, size_t index, bool odd_plane)
        : base_type(field, field->grid().xptr_celm(index, odd_plane, Grid::CelmPK()))
    {
    }

    class const_ctor_passkey
    {
        const_ctor_passkey() = default;
        friend Field;
    };

    Celm(Field const * field, size_t index, bool odd_plane, const_ctor_passkey)
        // The only intention of this workaround is to let const Field to
        // create const object derived from CelmBase. Do not abuse.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        : Celm(const_cast<Field *>(field), index, odd_plane)
    {
    }

    int_type index() const
    {
        static_assert(0 == (Grid::BOUND_COUNT % 2), "only work with even BOUND_COUNT");
        return (static_cast<int_type>(xindex() - 1) >> 1) - 1;
    }

    /**
     * Return true for even plane, false for odd plane (temporal).
     */
    bool on_even_plane() const { return !on_odd_plane(); }
    bool on_odd_plane() const { return static_cast<bool>((xindex() - (1 + Grid::BOUND_COUNT)) & 1); }

    value_type xctr() const { return x(); }

    void move_at(int_type offset);

    value_type time_increment() const { return field().time_increment(); }
    value_type dt() const { return field().dt(); }
    value_type hdt() const { return field().hdt(); }
    value_type qdt() const { return field().qdt(); }

    template <typename SE>
    SE const selm_xn() const { return field().template selm<SE>(index(), on_odd_plane()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_xn() { return field().template selm<SE>(index(), on_odd_plane()); }
    template <typename SE>
    SE const selm_xp() const { return field().template selm<SE>(index() + 1, on_odd_plane()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_xp() { return field().template selm<SE>(index() + 1, on_odd_plane()); }
    template <typename SE>
    SE const selm_tn() const { return field().template selm<SE>(index() + on_odd_plane(), !on_odd_plane()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_tn() { return field().template selm<SE>(index() + on_odd_plane(), !on_odd_plane()); }
    template <typename SE>
    SE const selm_tp() const { return field().template selm<SE>(index() + on_odd_plane(), !on_odd_plane()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_tp() { return field().template selm<SE>(index() + on_odd_plane(), !on_odd_plane()); }

    template <typename SE>
    value_type calc_so0(size_t iv) const;
    template <typename SE, size_t ALPHA>
    value_type calc_so1_alpha(size_t iv) const;

    Selm const selm_xn() const { return selm_xn<Selm>(); } // NOLINT(readability-const-return-type)
    Selm selm_xn() { return selm_xn<Selm>(); }
    Selm const selm_xp() const { return selm_xp<Selm>(); } // NOLINT(readability-const-return-type)
    Selm selm_xp() { return selm_xp<Selm>(); }
    Selm const selm_tn() const { return selm_tn<Selm>(); } // NOLINT(readability-const-return-type)
    Selm selm_tn() { return selm_tn<Selm>(); }
    Selm const selm_tp() const { return selm_tn<Selm>(); } // NOLINT(readability-const-return-type)
    Selm selm_tp() { return selm_tn<Selm>(); }

    value_type calc_so0(size_t iv) const { return calc_so0<Selm>(iv); }
    template <size_t ALPHA>
    value_type calc_so1_alpha(size_t iv) const { return calc_so1_alpha<Selm, ALPHA>(iv); }

}; /* end class Celm */

template <typename SE>
class CelmBase
    : public Celm
{

public:

    using base_type = Celm;
    using base_type::base_type;
    using selm_type = SE;

    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_xn() const { return this->Celm::selm_xn<SE>(); }
    SE selm_xn() { return this->Celm::selm_xn<SE>(); }
    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_xp() const { return this->Celm::selm_xp<SE>(); }
    SE selm_xp() { return this->Celm::selm_xp<SE>(); }
    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_tn() const { return this->Celm::selm_tn<SE>(); }
    SE selm_tn() { return this->Celm::selm_tn<SE>(); }
    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_tp() const { return this->Celm::selm_tp<SE>(); }
    SE selm_tp() { return this->Celm::selm_tp<SE>(); }

    value_type calc_so0(size_t iv) const { return this->Celm::calc_so0<SE>(iv); }
    template <size_t ALPHA>
    value_type calc_so1_alpha(size_t iv) const
    {
        return this->Celm::calc_so1_alpha<SE, ALPHA>(iv);
    }

}; /* end class CelmBase */

} /* end namespace spacetime */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
