/*
 * Copyright (c) 2016, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * Order-1 solution marching (gradient reconstruction) for the Euler CESE
 * solver.  calc_dsoln it builds a per-cell GradientElement, solves a gradient
 * on every fundamental gradient element, and reduces them with the W-1/2
 * weighting and W-3/4 limiter into the new order-1 solution so1n.
 */

#include <modmesh/multidim/euler.hpp>
#include <modmesh/multidim/GradientElement.hpp>

#include <array>
#include <cmath>

namespace modmesh
{

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void EulerCore::calc_dsoln()
{
    constexpr size_t NFGE_MAX = GradientElementType::NFGE_MAX;
    constexpr size_t NEQ_MAX = 5;
    // Floor that keeps the inverse weighting and the sigma_max bounds finite.
    constexpr real_type ALMOST_ZERO = 1.e-200;

    size_t const ndim = m_ndim;
    auto const neq = static_cast<size_t>(m_neq);
    real_type const hdt = m_time_increment * 0.5;
    auto const & msh = *m_mesh;

    for (int_type icl = 0; icl < m_ncell; ++icl)
    {
        // Per-cell weighting cap and gradient-element spread from the CFL.
        real_type const acfl = std::fabs(m_cflc(icl));
        real_type const sgm0 = m_sigma0 / acfl;
        real_type const tau = m_taumin + acfl * m_tauscale;

        GradientElement const gelem(msh, m_cecnd, icl, tau);
        int_type const nfge = gelem.nfge();
        real_type const ofg1 = gelem.nfge_inverse();

        std::array<std::array<GradientElement::ge_vector_type, NEQ_MAX>, NFGE_MAX> grad = {};
        std::array<std::array<real_type, NEQ_MAX>, NFGE_MAX> widv = {};
        std::array<real_type, NEQ_MAX> wacc = {};
        std::array<real_type, NEQ_MAX> sigma_max = {};

        // Per fundamental gradient element: interpolate the solution deltas,
        // solve the gradient, and accumulate the W-1/2 weighting.
        for (int_type ifge = 0; ifge < nfge; ++ifge)
        {
            GradientElementType::face_list_type const & tface = gelem.faces(ifge);
            // Solution deltas at the gradient evaluation points: udf[ieq][ivx].
            std::array<GradientElement::ge_vector_type, NEQ_MAX> udf = {};
            for (size_t ivx = 0; ivx < ndim; ++ivx)
            {
                int_type const ifl = tface[ivx] - 1;
                int_type const jcl = gelem.rcl(ifl);
                for (size_t ieq = 0; ieq < neq; ++ieq)
                {
                    // Taylor interpolation about the neighbor cell, relative to
                    // the self new-step solution.
                    real_type val = m_so0c(jcl, ieq) + hdt * m_so0t(jcl, ieq) - m_so0n(icl, ieq);
                    for (size_t d = 0; d < ndim; ++d)
                    {
                        val += gelem.jdis(ifl, static_cast<int_type>(d)) * m_so1c(jcl, ieq, d);
                    }
                    udf[ieq][ivx] = val;
                }
            }
            for (size_t ieq = 0; ieq < neq; ++ieq)
            {
                GradientElement::ge_vector_type const g = gelem.solve_gradient(ifge, udf[ieq]);
                grad[ifge][ieq] = g;
                real_type sq = 0.0;
                for (size_t d = 0; d < ndim; ++d)
                {
                    sq += g[d] * g[d];
                }
                // W-1/2 weighting (alpha = 1): inverse gradient magnitude.
                real_type const wgt = 1.0 / std::sqrt(sq + ALMOST_ZERO);
                wacc[ieq] += wgt;
                widv[ifge][ieq] = wgt;
            }
        }

        // W-3/4 limiter delta and the per-equation sigma_max cap.
        std::array<std::array<real_type, 2>, NEQ_MAX> wpa = {}; // {max, min}
        for (int_type ifge = 0; ifge < nfge; ++ifge)
        {
            for (size_t ieq = 0; ieq < neq; ++ieq)
            {
                real_type const wgt = widv[ifge][ieq] / wacc[ieq] - ofg1;
                widv[ifge][ieq] = wgt;
                wpa[ieq][0] = std::fmax(wpa[ieq][0], wgt);
                wpa[ieq][1] = std::fmin(wpa[ieq][1], wgt);
            }
        }
        for (size_t ieq = 0; ieq < neq; ++ieq)
        {
            real_type const sm = std::fmin(
                (1.0 - ofg1) / (wpa[ieq][0] + ALMOST_ZERO),
                -ofg1 / (wpa[ieq][1] - ALMOST_ZERO));
            sigma_max[ieq] = std::fmin(sm, sgm0);
        }

        // Weighted reduction of the per-element gradients into so1n.
        for (size_t ieq = 0; ieq < neq; ++ieq)
        {
            for (size_t d = 0; d < ndim; ++d)
            {
                m_so1n(icl, ieq, d) = 0.0;
            }
        }
        for (int_type isub = 0; isub < nfge; ++isub)
        {
            for (size_t ieq = 0; ieq < neq; ++ieq)
            {
                real_type const wgt = ofg1 + sigma_max[ieq] * widv[isub][ieq];
                for (size_t d = 0; d < ndim; ++d)
                {
                    m_so1n(icl, ieq, d) += wgt * grad[isub][ieq][d];
                }
            }
        }
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
