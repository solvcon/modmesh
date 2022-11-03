#pragma once

/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/spacetime/core.hpp>

namespace modmesh
{
namespace spacetime
{

class BadEuler1DSolver
    : public std::enable_shared_from_this<BadEuler1DSolver>
{

public:

    static constexpr uint8_t NVAR = 3;

    using celm_type = Celm;
    using selm_type = Selm;

private:

    struct ctor_passkey
    {
    };

public:

    std::shared_ptr<BadEuler1DSolver> clone(bool grid = false)
    {
        auto ret = std::make_shared<BadEuler1DSolver>(*this);
        if (grid)
        {
            std::shared_ptr<Grid> const new_grid = m_field.clone_grid();
            ret->m_field.set_grid(new_grid);
        }
        return ret;
    }

    template <class... Args>
    static std::shared_ptr<BadEuler1DSolver> construct(Args &&... args)
    {
        return std::make_shared<BadEuler1DSolver>(std::forward<Args>(args)..., ctor_passkey());
    }

    BadEuler1DSolver(
        std::shared_ptr<Grid> const & grid, double time_increment, ctor_passkey const &)
        : m_field(grid, time_increment, NVAR)
    {
    }

    explicit BadEuler1DSolver(ctor_passkey const &);

    BadEuler1DSolver() = delete;
    BadEuler1DSolver(BadEuler1DSolver const &) = default;
    BadEuler1DSolver(BadEuler1DSolver &&) = default;
    BadEuler1DSolver & operator=(BadEuler1DSolver const &) = default;
    BadEuler1DSolver & operator=(BadEuler1DSolver &&) = default;
    ~BadEuler1DSolver() = default;

    Field const & field() const { return m_field; }
    Field & field() { return m_field; }

    size_t nvar() const { return m_field.nvar(); }

    Grid const & grid() const { return m_field.grid(); }
    Grid & grid() { return m_field.grid(); }

    // NOLINTNEXTLINE(readability-const-return-type,cppcoreguidelines-pro-type-const-cast)
    Celm const celm(int_type ielm, bool odd_plane) const { return {const_cast<Field *>(&m_field), ielm, odd_plane}; }
    Celm celm(int_type ielm, bool odd_plane) { return {&m_field, ielm, odd_plane}; }
    // NOLINTNEXTLINE(readability-const-return-type,cppcoreguidelines-pro-type-const-cast)
    Selm const selm(int_type ielm, bool odd_plane) const { return {const_cast<Field *>(&m_field), ielm, odd_plane}; }
    Selm selm(int_type ielm, bool odd_plane) { return {&m_field, ielm, odd_plane}; }

    void update_cfl(bool odd_plane);
    void march_half_so0(bool odd_plane);
    template <size_t ALPHA>
    void march_half_so1_alpha(bool odd_plane);
    void treat_boundary_so0();
    void treat_boundary_so1();

    void setup_march() { update_cfl(false); }
    template <size_t ALPHA>
    void march_half1_alpha();
    template <size_t ALPHA>
    void march_half2_alpha();
    template <size_t ALPHA>
    void march_alpha(size_t steps);

private:

    Field m_field;
};

struct Euler1DKernel
{
    static constexpr double tiny = 1.e-100;

    Euler1DKernel() // NOLINT(cppcoreguidelines-pro-type-member-init)
    {
        jac[0][0] = 0.0;
        jac[0][1] = 1.0;
        jac[0][2] = 0.0;
    }

    Euler1DKernel & set_gamma(double gamma_)
    {
        gamma = gamma_;
        jac[1][2] = gamma - 1.0;
        return *this;
    }

    Euler1DKernel & set_time_increment(double time_increment)
    {
        hdt = time_increment / 2.0;
        qdt = hdt / 2.0;
        return *this;
    }

    Euler1DKernel & set_selm(Selm const & se)
    {
        x = se.x();
        xctr = se.xctr();
        xneg = se.xneg();
        xpos = se.xpos();
        u[0] = se.so0(0);
        u[1] = se.so0(1);
        u[2] = se.so0(2);
        ux[0] = se.so1(0);
        ux[1] = se.so1(1);
        ux[2] = se.so1(2);
        return *this;
    }

    Euler1DKernel & derive()
    {
        // TODO: reduce numerical caldulation.
        jac[1][0] = (gamma - 3.0) / 2.0 * u[1] * u[1] / (u[0] * u[0] + tiny);
        jac[1][1] = -(gamma - 3.0) * u[1] / (u[0] + tiny);
        jac[2][0] = (gamma - 1.0) * u[1] * u[1] * u[1] / (u[0] * u[0] * u[0] + tiny) - gamma * u[1] * u[2] / (u[0] * u[0] + tiny);
        jac[2][1] = gamma * u[2] / (u[0] + tiny) - 3.0 / 2.0 * (gamma - 1.0) * u[1] * u[1] / (u[0] * u[0] + tiny);
        jac[2][2] = gamma * u[1] / (u[0] + tiny);

        f[0] = u[1];
        f[1] = (gamma - 1.0) * u[2] + (3.0 - gamma) / 2.0 * u[1] * u[1] / (u[0] + tiny);
        f[2] = gamma * u[1] * u[2] / (u[0] + tiny) - (gamma - 1.0) / 2.0 * u[1] * u[1] * u[1] / (u[0] * u[0] + tiny);

        // Also ut = -fx
        ut[0] = -jac[0][0] * ux[0] - jac[0][1] * ux[1] - jac[0][2] * ux[2];
        ut[1] = -jac[1][0] * ux[0] - jac[1][1] * ux[1] - jac[1][2] * ux[2];
        ut[2] = -jac[2][0] * ux[0] - jac[2][1] * ux[1] - jac[2][2] * ux[2];

        ft[0] = jac[0][0] * ut[0] + jac[0][1] * ut[1] + jac[0][2] * ut[2];
        ft[1] = jac[1][0] * ut[0] + jac[1][1] * ut[1] + jac[1][2] * ut[2];
        ft[2] = jac[2][0] * ut[0] + jac[2][1] * ut[1] + jac[2][2] * ut[2];

        up[0] = u[0] + (x - xctr) * ux[0] /* displacement in x */ + hdt * ut[0] /* displacement in t */;
        up[1] = u[1] + (x - xctr) * ux[1] /* displacement in x */ + hdt * ut[1] /* displacement in t */;
        up[2] = u[2] + (x - xctr) * ux[2] /* displacement in x */ + hdt * ut[2] /* displacement in t */;

        return *this;
    }

    std::array<double, 3> calc_flux_ll()
    {
        const double deltax = xpos - x;
        const double offset1 = 0.5 * (x + xpos) - xctr;
        const double offset2 = x - xctr;
        std::array<double, 3> r; // NOLINT(cppcoreguidelines-pro-type-member-init)
        r[0] = deltax * (u[0] + offset1 * ux[0]) + hdt * (f[0] - (offset2 * ut[0]) + (qdt * ft[0]));
        r[1] = deltax * (u[1] + offset1 * ux[1]) + hdt * (f[1] - (offset2 * ut[1]) + (qdt * ft[1]));
        r[2] = deltax * (u[2] + offset1 * ux[2]) + hdt * (f[2] - (offset2 * ut[2]) + (qdt * ft[2]));
        return r;
    }

    std::array<double, 3> calc_flux_lr()
    {
        const double deltax = x - xneg;
        const double offset1 = 0.5 * (x + xneg) - xctr;
        const double offset2 = x - xctr;
        std::array<double, 3> r; // NOLINT(cppcoreguidelines-pro-type-member-init)
        r[0] = deltax * (u[0] + offset1 * ux[0]) - hdt * (f[0] - (offset2 * ut[0]) + (qdt * ft[0]));
        r[1] = deltax * (u[1] + offset1 * ux[1]) - hdt * (f[1] - (offset2 * ut[1]) + (qdt * ft[1]));
        r[2] = deltax * (u[2] + offset1 * ux[2]) - hdt * (f[2] - (offset2 * ut[2]) + (qdt * ft[2]));
        return r;
    }

    double gamma; //< Heat capacity ratio.
    double qdt; //< Quarter of time increment.
    double hdt; //< Half of time increment.
    double x; //< Grid point.
    double xctr; //< Solution point.
    double xneg; //< Left point of the solution element.
    double xpos; //< Right point of the solution element.
    std::array<double, 3> u; // Variable.
    std::array<double, 3> ux; // First-order derivative of u with respect to space.
    std::array<std::array<double, 3>, 3> jac; //< Jacobian.
    std::array<double, 3> f; //< Function.
    std::array<double, 3> ut; //< First-order derivative of u with respect to time.
    std::array<double, 3> ft; //< First-order derivative of f with respect to time.
    std::array<double, 3> up; //< Derived variable.
}; /* end struct Euler1DKernel */

template <size_t ALPHA>
inline void BadEuler1DSolver::march_half_so1_alpha(bool odd_plane)
{
    const int_type start = odd_plane ? -1 : 0;
    const int_type stop = static_cast<int_type>(grid().ncelm());
    const double gamma = 1.4;
    // Kernal at xneg solution element.
    Euler1DKernel kernxn;
    kernxn
        .set_gamma(gamma)
        .set_time_increment(m_field.dt());
    // Kernal at xpos solution element.
    Euler1DKernel kernxp;
    kernxp
        .set_gamma(gamma)
        .set_time_increment(m_field.dt())
        // Populate using the solution element.
        .set_selm(Selm(&m_field, start, odd_plane))
        .derive();
    for (int_type ic = start; ic < stop; ++ic)
    {
        // Update kernels (avoid duplicate expensiave calculation).
        kernxn = kernxp;
        kernxp
            .set_selm(Selm(&m_field, ic + 1, odd_plane))
            .derive();
        // Calculate the gradient.
        Selm se_tp(&m_field, ic, !odd_plane);
        for (size_t iv = 0; iv < 3; ++iv)
        {
            const double utp = se_tp.so0(iv);
            const double duxn = (utp - kernxn.up[iv]) / (kernxn.xpos - kernxn.x);
            const double duxp = (kernxp.up[iv] - utp) / (kernxp.x - kernxp.xneg);
            const double fan = pow<ALPHA>(std::abs(duxn));
            const double fap = pow<ALPHA>(std::abs(duxp));
            se_tp.so1(iv) = (fap * duxn + fan * duxp) / (fap + fan + Euler1DKernel::tiny);
        }
    }
}

template <size_t ALPHA>
inline void BadEuler1DSolver::march_half1_alpha()
{
    march_half_so0(false);
    treat_boundary_so0();
    update_cfl(true);
    march_half_so1_alpha<ALPHA>(false);
    treat_boundary_so1();
}

template <size_t ALPHA>
inline void BadEuler1DSolver::march_half2_alpha()
{
    // In the second half step, no treating boundary conditions.
    march_half_so0(true);
    update_cfl(false);
    march_half_so1_alpha<ALPHA>(true);
}

template <size_t ALPHA>
inline void BadEuler1DSolver::march_alpha(size_t steps)
{
    for (size_t it = 0; it < steps; ++it)
    {
        march_half1_alpha<ALPHA>();
        march_half2_alpha<ALPHA>();
    }
}

} /* end namespace spacetime */
} /* end namespace modmesh */