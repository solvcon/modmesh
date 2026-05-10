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

    explicit EigenSystem(array_type const & matrix);

    EigenSystem() = delete;
    EigenSystem(EigenSystem const &) = delete;
    EigenSystem(EigenSystem &&) = delete;
    EigenSystem & operator=(EigenSystem const &) = delete;
    EigenSystem & operator=(EigenSystem &&) = delete;
    ~EigenSystem() = default;

    /// Run DGEEV on the prepared workspace.
    void run();

    array_type const & matrix() const { return m_matrix; }
    array_type const & wr() const { return m_wr; }
    array_type const & wi() const { return m_wi; }
    array_type const & vl() const { return m_vl; }
    array_type const & vr() const { return m_vr; }
    bool done() const { return m_done; }

private:

    static std::string format_shape(array_type const & arr);

    SimpleArray<value_type> const & m_matrix;
    SimpleArray<value_type> m_colmajor;
    SimpleArray<value_type> m_wr;
    SimpleArray<value_type> m_wi;
    SimpleArray<value_type> m_vl;
    SimpleArray<value_type> m_vr;
    bool m_done = false;

}; /* end class EigenSystem */

inline EigenSystem::EigenSystem(array_type const & matrix)
    : m_matrix(matrix)
    , m_colmajor(matrix.shape())
    , m_wr(matrix.shape(0))
    , m_wi(matrix.shape(0))
    , m_vl(matrix.shape())
    , m_vr(matrix.shape())
{
    if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1))
    {
        throw std::invalid_argument(std::format(
            "EigenSystem: matrix must be a square 2D SimpleArray, but got shape {}",
            format_shape(matrix)));
    }

    // TODO: Enhance SimpleArray::transpose() to copy the contents when
    // transposing.  The current SimpleArray::transpose() is a *view-only*
    // operation: it reverses the shape and stride vectors but the underlying
    // buffer is untouched.  So we have to manually copy element-wisely in the
    // nested loops below.
    m_colmajor.transpose();
    ssize_t const n = matrix.shape(0);
    for (ssize_t i = 0; i < n; ++i)
    {
        for (ssize_t j = 0; j < n; ++j)
        {
            m_colmajor(i, j) = matrix(i, j);
        }
    }
    m_vl.transpose();
    m_vr.transpose();
}

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
    char const jobvl = 'V';
    char const jobvr = 'V';
    __LAPACK_int const lda = n;
    __LAPACK_int const ldvl = n;
    __LAPACK_int const ldvr = n;
    __LAPACK_int info = 0;

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
        m_vl.data(),
        &ldvl,
        m_vr.data(),
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

    // Phase 2: allocate workspace and run.  Floor at 4*n per the LAPACK
    // reference minimum when both eigenvector matrices are requested.
    lwork = std::max<__LAPACK_int>(static_cast<__LAPACK_int>(work_query), 4 * n);
    array_type work(static_cast<size_t>(lwork));
    dgeev_(
        &jobvl,
        &jobvr,
        &n,
        m_colmajor.data(),
        &lda,
        m_wr.data(),
        m_wi.data(),
        m_vl.data(),
        &ldvl,
        m_vr.data(),
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
