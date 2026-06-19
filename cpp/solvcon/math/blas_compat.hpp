#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/Complex.hpp>

#include <cstddef>

namespace solvcon
{

float dot_blas(size_t size, float const * lhs, float const * rhs);
double dot_blas(size_t size, double const * lhs, double const * rhs);
Complex<float> dot_blas(size_t size,
                        Complex<float> const * lhs,
                        Complex<float> const * rhs);
Complex<double> dot_blas(size_t size,
                         Complex<double> const * lhs,
                         Complex<double> const * rhs);
void gemv_blas(size_t m,
               size_t n,
               float const * matrix,
               float const * vector,
               float * result,
               bool transpose_matrix);
void gemv_blas(size_t m,
               size_t n,
               double const * matrix,
               double const * vector,
               double * result,
               bool transpose_matrix);
void gemv_blas(size_t m,
               size_t n,
               Complex<float> const * matrix,
               Complex<float> const * vector,
               Complex<float> * result,
               bool transpose_matrix);
void gemv_blas(size_t m,
               size_t n,
               Complex<double> const * matrix,
               Complex<double> const * vector,
               Complex<double> * result,
               bool transpose_matrix);
void gemm_blas(size_t m,
               size_t n,
               size_t k,
               float const * lhs,
               float const * rhs,
               float * result);
void gemm_blas(size_t m,
               size_t n,
               size_t k,
               double const * lhs,
               double const * rhs,
               double * result);
void gemm_blas(size_t m,
               size_t n,
               size_t k,
               Complex<float> const * lhs,
               Complex<float> const * rhs,
               Complex<float> * result);
void gemm_blas(size_t m,
               size_t n,
               size_t k,
               Complex<double> const * lhs,
               Complex<double> const * rhs,
               Complex<double> * result);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
