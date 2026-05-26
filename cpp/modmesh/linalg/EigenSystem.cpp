/*
 * Copyright (c) 2026, Yung-Yu Chen <yyc@solvcon.net>
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

#include <memory>
#include <stdexcept>
#include <variant>

// EigenSystem.hpp requires a vendor LAPACK; only compile the surrogate's
// out-of-line bodies when one is available.  Without it this is an empty TU.
#ifdef MM_HAS_VENDOR_LAPACK

#include <modmesh/linalg/EigenSystem.hpp>

namespace modmesh
{

// Reinterpret the plex's type-erased instance as the typed SimpleArray<T> it
// actually holds; the caller must have verified data_type() == T.
template <typename T>
static SimpleArray<T> const & typed_array(SimpleArrayPlex const & plex)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return *reinterpret_cast<SimpleArray<T> const *>(plex.instance_ptr());
}

EigenSystemPlex::EigenSystemPlex(SimpleArrayPlex const & matrix, bool do_vl, bool do_vr)
    : m_matrix(matrix)
    , m_data_type(matrix.data_type())
{
    switch (matrix.data_type())
    {
    case DataType::Float32:
        m_solver = EigenSystem<float>::construct(typed_array<float>(matrix), do_vl, do_vr);
        break;
    case DataType::Float64:
        m_solver = EigenSystem<double>::construct(typed_array<double>(matrix), do_vl, do_vr);
        break;
    case DataType::Complex64:
        m_solver = EigenSystem<Complex<float>>::construct(typed_array<Complex<float>>(matrix), do_vl, do_vr);
        break;
    case DataType::Complex128:
        m_solver = EigenSystem<Complex<double>>::construct(typed_array<Complex<double>>(matrix), do_vl, do_vr);
        break;
    default:
        throw std::invalid_argument(
            "EigenSystemPlex: SimpleArray data type must be float32, float64, complex64, or complex128");
    }
}

void EigenSystemPlex::run()
{
    std::visit([](auto const & solver)
               { solver->run(); },
               m_solver);
}

SimpleArrayPlex EigenSystemPlex::wr() const
{
    return std::visit(
        [](auto const & solver)
        { return SimpleArrayPlex(solver->wr()); },
        m_solver);
}

SimpleArrayPlex EigenSystemPlex::wi() const
{
    return std::visit(
        [](auto const & solver)
        { return SimpleArrayPlex(solver->wi()); },
        m_solver);
}

SimpleArrayPlex EigenSystemPlex::vl(bool suppress_exception) const
{
    return std::visit(
        [suppress_exception](auto const & solver)
        { return SimpleArrayPlex(solver->vl(suppress_exception)); },
        m_solver);
}

SimpleArrayPlex EigenSystemPlex::vr(bool suppress_exception) const
{
    return std::visit(
        [suppress_exception](auto const & solver)
        { return SimpleArrayPlex(solver->vr(suppress_exception)); },
        m_solver);
}

bool EigenSystemPlex::do_vl() const
{
    return std::visit([](auto const & solver)
                      { return solver->do_vl(); },
                      m_solver);
}

bool EigenSystemPlex::do_vr() const
{
    return std::visit([](auto const & solver)
                      { return solver->do_vr(); },
                      m_solver);
}

bool EigenSystemPlex::done() const
{
    return std::visit([](auto const & solver)
                      { return solver->done(); },
                      m_solver);
}

} /* end namespace modmesh */

#endif /* MM_HAS_VENDOR_LAPACK */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
