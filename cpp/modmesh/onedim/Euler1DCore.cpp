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

#include <modmesh/onedim/Euler1DCore.hpp>
#include <cmath>

namespace modmesh
{

namespace onedim
{

std::ostream & operator<<(std::ostream & os, const Euler1DCore & sol)
{
    os << "Euler1DCore(ncoord=" << sol.ncoord() << ", time_increment=" << sol.time_increment() << ")";
    return os;
}

void Euler1DCore::initialize_data(size_t ncoord)
{
    if (0 == ncoord % 2)
    {
        throw std::invalid_argument("ncoord cannot be even");
    }
    m_coord = SimpleArray<double>(/*length*/ ncoord);
    m_cfl = SimpleArray<double>(/*length*/ ncoord);
    m_so0 = SimpleArray<double>(/*shape*/ small_vector<size_t>{ncoord, NVAR});
    m_so1 = SimpleArray<double>(/*shape*/ small_vector<size_t>{ncoord, NVAR});
    m_gamma = SimpleArray<double>(/*shape*/ small_vector<size_t>{ncoord}, /*value*/ 1.4);
}

SimpleArray<double> Euler1DCore::density() const
{
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = density(it);
    }
    return ret;
}

SimpleArray<double> Euler1DCore::velocity() const
{
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = velocity(it);
    }
    return ret;
}

SimpleArray<double> Euler1DCore::pressure() const
{
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = pressure(it);
    }
    return ret;
}

void Euler1DCore::update_cfl(bool odd_plane)
{
    const int_type start = BOUND_COUNT - (odd_plane ? 1 : 0);
    const auto stop = static_cast<int_type>(ncoord() - BOUND_COUNT - (odd_plane ? 0 : 1));
    const double hdt = m_time_increment / 2;
    for (int_type it = start; it < stop; it += 2)
    {
        const double ga = m_gamma(it);
        // TODO: I didn't verify the formula.
        // wave speed.
        double wspd = m_so0(it, 1);
        wspd *= wspd;
        const double ke = wspd / (2.0 * m_so0(it, 0));
        double pr = (ga - 1.0) * (m_so0(it, 2) - ke);
        pr = (pr + std::abs(pr)) / 2.0;
        wspd = std::sqrt(ga * pr / m_so0(it, 0)) + std::sqrt(wspd) / m_so0(it, 0);
        // CFL.
        const double dxpos = m_coord(it + 1) - m_coord(it);
        const double dxneg = m_coord(it) - m_coord(it - 1);
        const double cfl = hdt * wspd / (dxpos < dxneg ? dxpos : dxneg);
        // Set back.
        m_cfl(it) = cfl;
    }
}

void Euler1DCore::march_half_so0(bool odd_plane)
{
    const int_type start = BOUND_COUNT - (odd_plane ? 1 : 0);
    const auto stop = static_cast<int_type>(ncoord() - BOUND_COUNT - (odd_plane ? 0 : 1));
    // Kernal at xneg solution element.
    Euler1DKernel kernxn{};
    kernxn
        .set_time_increment(m_time_increment);
    // Kernal at xpos solution element.
    Euler1DKernel kernxp{};
    kernxp
        .set_time_increment(m_time_increment)
        // Populate using the solution element.
        .set_value(start, m_gamma, m_coord, m_so0, m_so1)
        .derive();
    for (int_type ic = start; ic < stop; ic += 2)
    {
        // Update kernels (avoid duplicate expensiave calculation).
        kernxn = kernxp;
        kernxp
            .set_value(ic + 2, m_gamma, m_coord, m_so0, m_so1)
            .derive();
        // Calculate flux through the lower left and lower right of conservation element.
        const std::array<double, 3> flux_ll = kernxn.calc_flux_ll();
        const std::array<double, 3> flux_lr = kernxp.calc_flux_lr();
        // Calculate the variables using flux conservation.
        double const xneg = m_coord(ic);
        double const xpos = m_coord(ic + 2);
        double const dx = xpos - xneg;
        m_so0(ic + 1, 0) = (flux_ll[0] + flux_lr[0]) / dx;
        m_so0(ic + 1, 1) = (flux_ll[1] + flux_lr[1]) / dx;
        m_so0(ic + 1, 2) = (flux_ll[2] + flux_lr[2]) / dx;
    }
}

void Euler1DCore::treat_boundary_so0()
{
    // Set outside value from inside value.
    {
        // Left boundary.
        size_t const ic = 0;
        m_so0(ic, 0) = m_so0(ic + 2, 0);
        m_so0(ic, 1) = m_so0(ic + 2, 1);
        m_so0(ic, 2) = m_so0(ic + 2, 2);
    }
    {
        // Right boundary.
        size_t const ic = ncoord() - 1;
        m_so0(ic, 0) = m_so0(ic - 2, 0);
        m_so0(ic, 1) = m_so0(ic - 2, 1);
        m_so0(ic, 2) = m_so0(ic - 2, 2);
    }
}

void Euler1DCore::treat_boundary_so1()
{
    // Set outside value from inside value.
    {
        // Left boundary.
        size_t const ic = 0;
        m_so1(ic, 0) = m_so1(ic + 2, 0);
        m_so1(ic, 1) = m_so1(ic + 2, 1);
        m_so1(ic, 2) = m_so1(ic + 2, 2);
    }
    {
        // Right boundary.
        size_t const ic = ncoord() - 1;
        m_so1(ic, 0) = m_so1(ic - 2, 0);
        m_so1(ic, 1) = m_so1(ic - 2, 1);
        m_so1(ic, 2) = m_so1(ic - 2, 2);
    }
}

} /* end namespace onedim */
} /* end namespace modmesh */
