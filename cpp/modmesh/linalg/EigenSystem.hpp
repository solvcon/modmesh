#pragma once

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

/**
 * Eigenvalue and eigenvector computation for general (non-symmetric) real
 * matrices using LAPACK DGEEV from Apple's vecLib (Accelerate framework).
 *
 * This header is only available on Apple platforms.
 */

#ifndef __APPLE__
#error "modmesh/linalg/EigenSystem.hpp is only available on Apple platforms (Accelerate/vecLib)."
#endif

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

#include <modmesh/buffer/buffer.hpp>

// Opt in to the modern (non-deprecated) LAPACK signatures provided by
// Accelerate.  Must be defined before including the Accelerate header.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
// NOLINTNEXTLINE(misc-header-include-cycle)
#include <Accelerate/Accelerate.h>

namespace modmesh
{

/**
 * Eigenvalue solver for a real general matrix using LAPACK DGEEV.
 *
 * Construction validates the input shape and prepares column-major workspace
 * buffers.  Call run() to invoke DGEEV to calculate eigenvalues and
 * eigenvectors.
 */
class EigenSystem
{

public:

    using value_type = double;
    using array_type = SimpleArray<double>;

    explicit EigenSystem(array_type const & matrix, bool do_vl = true, bool do_vr = true);

    EigenSystem() = delete;
    EigenSystem(EigenSystem const &) = delete;
    EigenSystem(EigenSystem &&) = delete;
    EigenSystem & operator=(EigenSystem const &) = delete;
    EigenSystem & operator=(EigenSystem &&) = delete;
    ~EigenSystem() = default;

    void run();

    array_type const & matrix() const { return m_matrix; }
    array_type const & wr() const { return m_wr; }
    array_type const & wi() const { return m_wi; }
    array_type const & vl(bool suppress_exception = false) const;
    array_type const & vr(bool suppress_exception = false) const;
    bool do_vl() const { return m_do_vl; }
    bool do_vr() const { return m_do_vr; }
    bool done() const { return m_done; }

private:

    static std::string format_shape(array_type const & arr);

    SimpleArray<value_type> const & m_matrix;
    SimpleArray<value_type> m_colmajor;
    SimpleArray<value_type> m_wr;
    SimpleArray<value_type> m_wi;
    SimpleArray<value_type> m_vl;
    SimpleArray<value_type> m_vr;
    bool const m_do_vl;
    bool const m_do_vr;
    bool m_done = false;

}; /* end class EigenSystem */

inline EigenSystem::EigenSystem(array_type const & matrix, bool do_vl, bool do_vr)
    : m_matrix(matrix)
    // Stage matrix into a column-major workspace for LAPACK.
    , m_colmajor(matrix.to_column_major())
    , m_wr(matrix.shape(0))
    , m_wi(matrix.shape(0))
    , m_vl(do_vl ? array_type(matrix.shape()) : array_type())
    , m_vr(do_vr ? array_type(matrix.shape()) : array_type())
    , m_do_vl(do_vl)
    , m_do_vr(do_vr)
{
    if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1))
    {
        throw std::invalid_argument(std::format(
            "EigenSystem: matrix must be a square 2D SimpleArray, but got shape {}",
            format_shape(matrix)));
    }

    if (m_do_vl)
    {
        m_vl.transpose();
    }
    if (m_do_vr)
    {
        m_vr.transpose();
    }
}

/**
 * @brief Run DGEEV on the prepared workspace.
 */
inline void EigenSystem::run()
{
    auto const n = static_cast<__LAPACK_int>(m_matrix.shape(0));
    if (n == 0)
    {
        m_done = true;
        return;
    }

    /*
     * Apple Accelerate DGEEV reference:
     *   https://developer.apple.com/documentation/accelerate/dgeev_(_:_:_:_:_:_:_:_:_:_:_:_:_:_:)
     * LAPACK DGEEV API reference:
     *   https://www.netlib.org/lapack/explore-html/d4/d68/group__geev_ga7d8afe93d23c5862e238626905ee145e.html
     *   https://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f_source.html
     */
    char const jobvl = m_do_vl ? 'V' : 'N';
    char const jobvr = m_do_vr ? 'V' : 'N';
    __LAPACK_int const lda = n;
    // DGEEV requires LDVL/LDVR >= 1 and a valid (non-null) pointer even
    // when the matrix is unreferenced; route the unused side to a stack
    // scratch slot.
    __LAPACK_int const ldvl = m_do_vl ? n : 1;
    __LAPACK_int const ldvr = m_do_vr ? n : 1;
    __LAPACK_int info = 0;
    double vl_dummy = 0.0;
    double vr_dummy = 0.0;
    double * const vl_ptr = m_do_vl ? m_vl.data() : &vl_dummy;
    double * const vr_ptr = m_do_vr ? m_vr.data() : &vr_dummy;

    // Phase 1: workspace query.  lwork == -1 tells DGEEV to write the
    // optimal workspace size into work[0] without performing any work.
    double work_query = 0.0;
    __LAPACK_int lwork = -1;
    dgeev_(
        &jobvl,
        &jobvr,
        &n,
        m_colmajor.data(),
        &lda,
        m_wr.data(),
        m_wi.data(),
        vl_ptr,
        &ldvl,
        vr_ptr,
        &ldvr,
        &work_query,
        &lwork,
        &info);
    if (info != 0)
    {
        throw std::runtime_error(std::format(
            "EigenSystem::run: DGEEV workspace query failed with info={}",
            static_cast<int64_t>(info)));
    }

    // Phase 2: 4*n minimum when any eigenvectors requested, else 3*n.
    __LAPACK_int const lwork_min = (m_do_vl || m_do_vr) ? 4 * n : 3 * n;
    lwork = std::max<__LAPACK_int>(static_cast<__LAPACK_int>(work_query), lwork_min);
    array_type work(static_cast<size_t>(lwork));
    dgeev_(
        &jobvl,
        &jobvr,
        &n,
        m_colmajor.data(),
        &lda,
        m_wr.data(),
        m_wi.data(),
        vl_ptr,
        &ldvl,
        vr_ptr,
        &ldvr,
        work.data(),
        &lwork,
        &info);
    if (info < 0)
    {
        throw std::runtime_error(std::format(
            "EigenSystem::run: DGEEV illegal argument at position {}",
            -static_cast<int>(info)));
    }
    if (info > 0)
    {
        throw std::runtime_error(std::format(
            "EigenSystem::run: DGEEV failed to converge; {} eigenvalues did not converge",
            static_cast<int>(info)));
    }
    m_done = true;
}

/**
 * @brief Left eigenvectors computed by DGEEV.
 *
 * @param suppress_exception
 *      Return the empty placeholder instead of throwing when the matrix was
 *      not computed.
 * @return
 *      Column-major n-by-n matrix; empty when do_vl=false and
 *      suppress_exception=true.
 * @throws std::runtime_error When do_vl=false and suppress_exception=false.
 */
inline EigenSystem::array_type const & EigenSystem::vl(bool suppress_exception) const
{
    if (!m_do_vl && !suppress_exception)
    {
        throw std::runtime_error(
            "EigenSystem::vl: left eigenvectors were not computed "
            "(do_vl=false)");
    }
    return m_vl;
}

/**
 * @brief Right eigenvectors computed by DGEEV.
 *
 * @param suppress_exception
 *      Return the empty placeholder instead of throwing when the matrix was
 *      not computed.
 * @return
 *      Column-major n-by-n matrix; empty when do_vr=false and
 *      suppress_exception=true.
 * @throws std::runtime_error When do_vr=false and suppress_exception=false.
 */
inline EigenSystem::array_type const & EigenSystem::vr(bool suppress_exception) const
{
    if (!m_do_vr && !suppress_exception)
    {
        throw std::runtime_error(
            "EigenSystem::vr: right eigenvectors were not computed "
            "(do_vr=false)");
    }
    return m_vr;
}

inline std::string EigenSystem::format_shape(array_type const & arr)
{
    std::string result = "(";
    for (size_t i = 0; i < arr.ndim(); ++i)
    {
        if (i > 0)
        {
            result += ", ";
        }
        result += std::to_string(arr.shape(i));
    }
    result += ")";
    return result;
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
