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
 * Compatibility shim for the vendor LAPACK backend.
 *
 * On Apple, pull in Accelerate/vecLib with the modern (non-deprecated) LAPACK
 * signatures.  On other platforms (Linux + OpenBLAS), declare the Fortran ABI
 * symbols directly; this avoids pulling in lapacke.h, which Linux
 * distributions package inconsistently.
 *
 * Consumers should include this header and use `lapack_int_t` plus the
 * Fortran-named entry points (e.g. `dgeev_`) without conditional compilation.
 * The header itself requires `MM_HAS_VENDOR_LAPACK` to be defined by the build
 * system.
 */

#ifndef MM_HAS_VENDOR_LAPACK
#error "modmesh/linalg/lapack_compat.hpp requires a vendor LAPACK (MM_HAS_VENDOR_LAPACK)."
#endif

#ifdef __APPLE__
// Opt in to the modern (non-deprecated) LAPACK signatures provided by
// Accelerate.  Must be defined before including the Accelerate header.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
// NOLINTNEXTLINE(misc-header-include-cycle)
#include <Accelerate/Accelerate.h>
#else // __APPLE__
extern "C" void dgeev_(
    char const * jobvl,
    char const * jobvr,
    int const * n,
    double * a,
    int const * lda,
    double * wr,
    double * wi,
    double * vl,
    int const * ldvl,
    double * vr,
    int const * ldvr,
    double * work,
    int const * lwork,
    int * info);
#endif // __APPLE__

namespace modmesh
{

#ifdef __APPLE__
using lapack_int_t = __LAPACK_int;
#else
using lapack_int_t = int;
#endif

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
