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

#include <modmesh/spacetime/kernel/BadEuler1DSolver.hpp>
#include <cmath>

namespace modmesh
{
namespace spacetime
{

void BadEuler1DSolver::update_cfl(bool odd_plane)
{
    const int_type start = odd_plane ? -1 : 0;
    const int_type stop = static_cast<int_type>(grid().nselm());
    // TODO: specific heat ratio should not be hard-coded.
    constexpr double ga = 1.4;
    constexpr double ga1 = ga - 1.0;
    for (int_type ic = start; ic < stop; ++ic)
    {
        // TODO: I didn't verify the formula.
        Selm se(&m_field, ic, odd_plane);
        // wave speed.
        double wspd = se.so0(1) * se.so0(1);
        const double ke = wspd / (2.0 * se.so0(0));
        double pr = ga1 * (se.so0(2) - ke);
        pr = (pr + std::abs(pr)) / 2.0;
        wspd = std::sqrt(ga * pr / se.so0(0)) + std::sqrt(wspd) / se.so0(0);
        // CFL.
        const double dxpos = se.xpos() - se.x();
        const double dxneg = se.x() - se.xneg();
        const double cfl = m_field.hdt() * wspd / (dxpos < dxneg ? dxpos : dxneg);
        // Set back.
        se.cfl() = cfl;
    }
}

void BadEuler1DSolver::march_half_so0(bool odd_plane)
{
    const int_type start = odd_plane ? -1 : 0;
    const auto stop = static_cast<int_type>(grid().ncelm());
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
        // Calculate flux through the lower left and lower right of conservation element.
        const std::array<double, 3> flux_ll = kernxn.calc_flux_ll();
        const std::array<double, 3> flux_lr = kernxp.calc_flux_lr();
        // Calculate the variables using flux conservation.
        Selm se_tp(&m_field, ic, !odd_plane);
        const double dx = se_tp.dx();
        se_tp.so0(0) = (flux_ll[0] + flux_lr[0]) / dx;
        se_tp.so0(1) = (flux_ll[1] + flux_lr[1]) / dx;
        se_tp.so0(2) = (flux_ll[2] + flux_lr[2]) / dx;
    }
}

void BadEuler1DSolver::treat_boundary_so0()
{
    selm_type const selm_left_in = selm(0, true);
    selm_type selm_left_out = selm(-1, true);
    selm_type const selm_right_in = selm(static_cast<int_type>(grid().ncelm()) - 1, true);
    selm_type selm_right_out = selm(static_cast<int_type>(grid().ncelm()), true);

    // Periodic boundary condition treatment.
    selm_left_out.so0(0) = selm_right_in.so0(0);
    selm_left_out.so0(1) = selm_right_in.so0(1);
    selm_left_out.so0(2) = selm_right_in.so0(2);
    selm_right_out.so0(0) = selm_left_in.so0(0);
    selm_right_out.so0(1) = selm_left_in.so0(1);
    selm_right_out.so0(2) = selm_left_in.so0(2);
}

void BadEuler1DSolver::treat_boundary_so1()
{
    selm_type const selm_left_in = selm(0, true);
    selm_type selm_left_out = selm(-1, true);
    selm_type const selm_right_in = selm(static_cast<int_type>(grid().ncelm()) - 1, true);
    selm_type selm_right_out = selm(static_cast<int_type>(grid().ncelm()), true);

    // Periodic boundary condition treatment.
    selm_left_out.so1(0) = selm_right_in.so1(0);
    selm_left_out.so1(1) = selm_right_in.so1(1);
    selm_left_out.so1(2) = selm_right_in.so1(2);
    selm_right_out.so1(0) = selm_left_in.so1(0);
    selm_right_out.so1(1) = selm_left_in.so1(1);
    selm_right_out.so1(2) = selm_left_in.so1(2);
}

} /* end namespace spacetime */
} /* end namespace modmesh */
