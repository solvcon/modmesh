#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>
#include <vector>
#include <functional>

#include <modmesh/spacetime/system.hpp>
#include <modmesh/spacetime/type.hpp>
#include <modmesh/spacetime/Grid_decl.hpp>

namespace spacetime
{

class Celm;
class Selm;

/**
 * Calculation kernel for the physical problem to be solved.  The kernel
 * defines how the solution elements calculate fluxes and other values.
 */
class Kernel
{

public:

    using value_type = Grid::value_type;

    using calc_type1 = std::function<value_type(Selm const &, size_t)>;
    using calc_type2 = std::function<void(Selm &)>;

    Kernel() { reset(); }
    void reset();

    Kernel(Kernel const &) = default;
    Kernel(Kernel &&) = default;
    Kernel & operator=(Kernel const &) = default;
    Kernel & operator=(Kernel &&) = default;
    ~Kernel() = default;

    // Accessors.
    calc_type1 const & xn_calc() const { return m_xn_calc; }
    calc_type1 & xn_calc() { return m_xn_calc; }
    calc_type1 const & xp_calc() const { return m_xp_calc; }
    calc_type1 & xp_calc() { return m_xp_calc; }
    calc_type1 const & tn_calc() const { return m_tn_calc; }
    calc_type1 & tn_calc() { return m_tn_calc; }
    calc_type1 const & tp_calc() const { return m_tp_calc; }
    calc_type1 & tp_calc() { return m_tp_calc; }
    calc_type1 const & so0p_calc() const { return m_so0p_calc; }
    calc_type1 & so0p_calc() { return m_so0p_calc; }
    calc_type2 const & cfl_updater() const { return m_cfl_updater; }
    calc_type2 & cfl_updater() { return m_cfl_updater; }

    // Calculating functions.
    value_type calc_xn(Selm const & se, size_t iv) const { return m_xn_calc(se, iv); }
    value_type calc_xp(Selm const & se, size_t iv) const { return m_xp_calc(se, iv); }
    value_type calc_tn(Selm const & se, size_t iv) const { return m_tn_calc(se, iv); }
    value_type calc_tp(Selm const & se, size_t iv) const { return m_tp_calc(se, iv); }
    value_type calc_so0p(Selm const & se, size_t iv) const { return m_so0p_calc(se, iv); }
    void update_cfl(Selm & se) { m_cfl_updater(se); }

private:

    calc_type1 m_xn_calc;
    calc_type1 m_xp_calc;
    calc_type1 m_tn_calc;
    calc_type1 m_tp_calc;
    calc_type1 m_so0p_calc;
    calc_type2 m_cfl_updater;

}; /* end class Kernel */

/**
 * Data class for solution.  It doesn't contain type information for the CE and
 * SE.  A Field declared as const is useless.
 */
class Field
{

public:

    using value_type = Grid::value_type;
    using array_type = Grid::array_type;

    Field(std::shared_ptr<Grid> const & grid, value_type time_increment, size_t nvar);

    Field() = delete;
    Field(Field const &) = default;
    Field(Field &&) = default;
    Field & operator=(Field const &) = default;
    Field & operator=(Field &&) = default;
    ~Field() = default;

    Field clone(bool grid = false) const
    {
        Field ret(*this);
        if (grid)
        {
            ret.m_grid = clone_grid();
        }
        return ret;
    }

    std::shared_ptr<Grid> clone_grid() const
    {
        return m_grid->clone();
    }

    void set_grid(std::shared_ptr<Grid> const & grid) { m_grid = grid; }

    Grid const & grid() const { return *m_grid; }
    Grid & grid() { return *m_grid; }

    array_type const & so0() const { return m_so0; }
    array_type & so0() { return m_so0; }
    array_type const & so1() const { return m_so1; }
    array_type & so1() { return m_so1; }
    array_type const & cfl() const { return m_cfl; }
    array_type & cfl() { return m_cfl; }

    value_type const & so0(size_t it, size_t iv) const { return m_so0(it, iv); }
    value_type & so0(size_t it, size_t iv) { return m_so0(it, iv); }
    value_type const & so1(size_t it, size_t iv) const { return m_so1(it, iv); }
    value_type & so1(size_t it, size_t iv) { return m_so1(it, iv); }
    value_type const & cfl(size_t it) const { return m_cfl(it); }
    value_type & cfl(size_t it) { return m_cfl(it); }

    size_t nvar() const { return m_so0.shape()[1]; }

    void set_time_increment(value_type time_increment);

    real_type time_increment() const { return m_time_increment; }
    real_type dt() const { return m_time_increment; }
    real_type hdt() const { return m_half_time_increment; }
    real_type qdt() const { return m_quarter_time_increment; }

    Kernel const & kernel() const { return m_kernel; }
    Kernel & kernel() { return m_kernel; }

    template <typename CE>
    CE const celm(int_type ielm, bool odd_plane) const { return CE(this, ielm, odd_plane, typename CE::const_ctor_passkey()); } // NOLINT(readability-const-return-type)
    template <typename CE>
    CE celm(int_type ielm, bool odd_plane) { return CE(this, ielm, odd_plane); }
    template <typename CE>
    CE const celm_at(int_type ielm, bool odd_plane) const; // NOLINT(readability-const-return-type)
    template <typename CE>
    CE celm_at(int_type ielm, bool odd_plane);

    template <typename SE>
    SE const selm(int_type ielm, bool odd_plane) const { return SE(this, ielm, odd_plane, typename SE::const_ctor_passkey()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm(int_type ielm, bool odd_plane) { return SE(this, ielm, odd_plane); }
    template <typename SE>
    SE const selm_at(int_type ielm, bool odd_plane) const; // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_at(int_type ielm, bool odd_plane);

private:

    std::shared_ptr<Grid> m_grid;

    array_type m_so0;
    array_type m_so1;
    array_type m_cfl;

    real_type m_time_increment = 0;
    // Cached value;
    real_type m_half_time_increment = 0;
    real_type m_quarter_time_increment = 0;

    Kernel m_kernel;

}; /* end class Field */

} /* end namespace spacetime */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
