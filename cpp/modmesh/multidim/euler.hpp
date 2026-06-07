#pragma once

/*
 * Copyright (c) 2024, Yung-Yu Chen <yyc@solvcon.net>
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

/**
 * The space-time CESE solver for the Euler equation.
 */

#include <modmesh/mesh/mesh.hpp>

#include <array>

namespace modmesh
{

class EulerCore
    : public NumberBase<int32_t, double>
    , public std::enable_shared_from_this<EulerCore>
{

private:

    class ctor_passkey
    {
    };

public:

    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

    template <typename... Args>
    static std::shared_ptr<EulerCore> construct(Args &&... args)
    {
        return std::make_shared<EulerCore>(std::forward<Args>(args)..., ctor_passkey());
    }

    EulerCore(std::shared_ptr<StaticMesh> const & mesh, real_type time_increment, ctor_passkey const &);

    EulerCore() = delete;
    EulerCore(EulerCore const &) = delete;
    EulerCore(EulerCore &&) = delete;
    EulerCore & operator=(EulerCore const &) = delete;
    EulerCore & operator=(EulerCore &&) = delete;
    ~EulerCore() = default;

    std::shared_ptr<StaticMesh> const & mesh() const { return m_mesh; }
    uint8_t ndim() const { return m_ndim; }
    int_type ncell() const { return m_ncell; }
    int_type ngstcell() const { return m_ngstcell; }
    // Number of conserved equations: density + ndim momentum + total energy.
    int_type neq() const { return m_neq; }
    real_type time_increment() const { return m_time_increment; }

    real_type sigma0() const { return m_sigma0; }
    real_type taumin() const { return m_taumin; }
    real_type tauscale() const { return m_tauscale; }
    void set_sigma0(real_type v) { m_sigma0 = v; }
    void set_taumin(real_type v) { m_taumin = v; }
    void set_tauscale(real_type v) { m_tauscale = v; }

    SimpleArray<real_type> & cevol() { return m_cevol; }
    SimpleArray<real_type> & cecnd() { return m_cecnd; }
    SimpleArray<real_type> & sfcnd() { return m_sfcnd; }
    SimpleArray<real_type> & sfnml() { return m_sfnml; }

    SimpleArray<real_type> & so0c() { return m_so0c; }
    SimpleArray<real_type> & so0n() { return m_so0n; }
    SimpleArray<real_type> & so0t() { return m_so0t; }
    SimpleArray<real_type> & so1c() { return m_so1c; }
    SimpleArray<real_type> & so1n() { return m_so1n; }
    SimpleArray<real_type> & stm() { return m_stm; }
    SimpleArray<real_type> & cflo() { return m_cflo; }
    SimpleArray<real_type> & cflc() { return m_cflc; }
    SimpleArray<real_type> & gamma() { return m_gamma; }

    void prepare_ce();

    void init_solution(real_type gamma, real_type rho, std::array<real_type, 3> const & velocity, real_type p);
    void calc_cfl();
    void update()
    {
        m_so0c.swap(m_so0n);
        m_so1c.swap(m_so1n);
    }

    // Solution marching: calc_solt, calc_soln, calc_dsoln.

    // calc_solt fills the order-0 temporal derivative so0t from the Jacobian
    // and so1c.
    void calc_solt();
    // calc_soln advances the order-0 solution so0n with the CESE flux integral
    // over the BCEs.
    void calc_soln();
    // calc_dsoln reconstructs the order-1 derivative so1n from a per-cell
    // GradientElement plus the gradient weighting
    void calc_dsoln();

    // march_substep runs one CESE substep and march runs the requested number
    // of full steps (SUBSTEP_RUN substeps per step).
    void march_substep();
    void march(int_type steps);

    // Number of CESE substeps per full marching step.
    static constexpr int_type SUBSTEP_RUN = 2;

private:

    void initialize_arrays();
    void initialize_solution();

    void prepare_ce_2d();
    void prepare_ce_3d();

    std::shared_ptr<StaticMesh> m_mesh;
    real_type m_time_increment = 0.0;

    uint8_t m_ndim = 0;
    int_type m_ncell = 0;
    int_type m_ngstcell = 0;
    int_type m_neq = 0;

    // sigma0 caps the gradient weighting
    real_type m_sigma0 = 3.0;
    // tau (taumin + |cfl| * tauscale) sets the per-cell gradient-element
    // spread.
    real_type m_taumin = 0.0;
    real_type m_tauscale = 1.0;

    // CE geometry arrays.
    SimpleArray<real_type> m_cevol; // [ncell, CLMFC+1]
    SimpleArray<real_type> m_cecnd; // [ncell, (CLMFC+1)*ndim]
    SimpleArray<real_type> m_sfcnd; // [ncell, CLMFC*FCMND, ndim]
    SimpleArray<real_type> m_sfnml; // [ncell, CLMFC*FCMND, ndim]

    // Solution arrays.  The first axis spans ghost + body cells (set_nghost),
    // so every table carries ghost rows; neq = ndim + 2.
    SimpleArray<real_type> m_so0c; // [total, neq] order-0, current step
    SimpleArray<real_type> m_so0n; // [total, neq] order-0, new step
    SimpleArray<real_type> m_so0t; // [total, neq] order-0, temporal
    SimpleArray<real_type> m_so1c; // [total, neq, ndim] order-1, current step
    SimpleArray<real_type> m_so1n; // [total, neq, ndim] order-1, new step
    SimpleArray<real_type> m_stm; // [total, neq] solution time-marching scratch
    SimpleArray<real_type> m_cflo; // [total] original CFL number
    SimpleArray<real_type> m_cflc; // [total] clamped CFL number
    SimpleArray<real_type> m_gamma; // [total] ratio of specific heat

}; /* end class EulerCore */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
