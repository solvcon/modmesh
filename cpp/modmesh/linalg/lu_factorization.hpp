#pragma once

/*
 * Copyright (c) 2026, Anchi Liu <phy.tiger@gmail.com>
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

#include <algorithm>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <modmesh/buffer/buffer.hpp>

namespace modmesh
{

namespace detail
{

/**
 * LU decomposition with partial pivoting for general (non-symmetric) matrices.
 *
 * Given an n-by-n matrix A, this class computes the factorization PA = LU,
 * where:
 *   - P is a permutation matrix (represented as a pivot index vector),
 *   - L is a unit lower triangular matrix (diagonal elements are implicitly 1),
 *   - U is an upper triangular matrix.
 *
 * L and U are stored compactly in a single n-by-n array: the strictly lower
 * triangle holds L (excluding the unit diagonal), and the upper triangle
 * (including the diagonal) holds U.
 *
 * The pivot vector piv[k] records that row k was swapped with row piv[k]
 * during the k-th elimination step.  Swaps are applied sequentially from
 * k = 0 to k = n-1.
 *
 * Supported element types: float, double, Complex<float>, Complex<double>.
 */
template <typename T>
class Lu
{

    static_assert(
        is_real_v<T> || is_complex_v<T>,
        "Lu<T> requires T to be a real or complex number type");

private:

    using value_type = T;
    using real_type = typename detail::select_real_t<value_type>::type;
    using array_type = SimpleArray<value_type>;

    // Helper: format a SimpleArray shape as "(d0, d1, ...)" for error messages.
    static std::string format_shape(array_type const & arr);

public:

    /**
     * Compute the LU factorization with partial pivoting: PA = LU.
     *
     * @param a  Square 2D SimpleArray (the matrix to factorize).
     * @return   A pair of (lu_matrix, pivot_vector):
     *           - lu_matrix: n-by-n array with L in the strictly lower triangle
     *             and U in the upper triangle (including diagonal).
     *           - pivot_vector: length-n vector where piv[k] is the row index
     *             that was swapped with row k at step k.
     * @throws std::invalid_argument if a is not a square 2D array.
     * @throws std::runtime_error    if the matrix is singular or near-singular.
     */
    static std::pair<array_type, std::vector<ssize_t>> factorize(array_type const & a);

    /**
     * Solve the linear system Ax = b using LU factorization.
     *
     * @param a  Square 2D SimpleArray (the coefficient matrix).
     * @param b  1D or 2D SimpleArray (the right-hand side).
     *           If 1D, it is treated as a single column vector.
     *           If 2D with shape (n, m), each column is a separate RHS.
     * @return   The solution x with the same shape as b.
     * @throws std::invalid_argument if a is not square, b has wrong rank, or
     *         dimensions are incompatible.
     * @throws std::runtime_error    if a is singular.
     */
    static array_type solve(array_type const & a, array_type const & b);

    /**
     * Compute the matrix inverse A^(-1).
     *
     * Internally solves AX = I where I is the identity matrix.
     *
     * @param a  Square 2D SimpleArray.
     * @return   The inverse matrix A^(-1) with the same shape as a.
     * @throws std::invalid_argument if a is not a square 2D array.
     * @throws std::runtime_error    if a is singular.
     */
    static array_type inv(array_type const & a);

private:

    /**
     * Forward substitution: solve Ly = Pb.
     *
     * Applies the row permutation recorded in piv to b, then solves the
     * unit lower triangular system.  L is stored in the strictly lower
     * triangle of the combined lu matrix (diagonal of L is implicitly 1).
     *
     * @param lu   Combined LU matrix from factorize().
     * @param piv  Pivot vector from factorize().
     * @param b    2D RHS matrix (already reshaped if originally 1D).
     * @return     The intermediate result y.
     */
    static array_type forward_substitution(array_type const & lu, std::vector<ssize_t> const & piv, array_type const & b);

    /**
     * Backward substitution: solve Ux = y.
     *
     * U is stored in the upper triangle (including diagonal) of the
     * combined lu matrix.
     *
     * @param lu  Combined LU matrix from factorize().
     * @param y   The result of forward_substitution().
     * @return    The final solution x.
     */
    static array_type backward_substitution(array_type const & lu, array_type const & y);

}; /* end class Lu */

template <typename T>
auto Lu<T>::format_shape(array_type const & arr) -> std::string
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

template <typename T>
auto Lu<T>::factorize(array_type const & a) -> std::pair<array_type, std::vector<ssize_t>>
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1))
    {
        throw std::invalid_argument(std::format(
            "Lu::factorize: a must be a square 2D SimpleArray, but got shape {}",
            format_shape(a)));
    }

    const auto n = static_cast<ssize_t>(a.shape(0));

    // Create a working copy of the input matrix.  The algorithm modifies
    // this array in-place, overwriting it with the combined L and U factors.
    auto lu = array_type(a.shape(), value_type{0});
    std::copy(a.begin(), a.end(), lu.begin());

    // piv[k] records which row was swapped with row k at step k.
    // Initialize with std::iota so piv[k] = k (identity permutation).
    std::vector<ssize_t> piv(n);
    std::iota(piv.begin(), piv.end(), ssize_t{0});

    // Main elimination loop.  For each column k, we:
    //   1. Search column k (rows k..n-1) for the largest absolute value
    //      (partial pivoting) to improve numerical stability.
    //   2. Swap the pivot row with row k.
    //   3. Check for singularity.
    //   4. Compute multipliers and update the trailing submatrix.
    for (ssize_t k = 0; k < n; ++k)
    {
        // Step 1: Find the pivot -- the row in [k, n) with the largest
        // absolute value in column k.  Note: abs() returns real_type even
        // for complex value_type, so all comparison variables must be real.
        real_type max_val = abs(lu(k, k));
        ssize_t max_row = k;
        for (ssize_t i = k + 1; i < n; ++i)
        {
            real_type val = abs(lu(i, k));
            if (val > max_val)
            {
                max_val = val;
                max_row = i;
            }
        }
        piv[k] = max_row;

        // Step 2: Swap rows k and max_row in the full working matrix.
        if (max_row != k)
        {
            for (ssize_t j = 0; j < n; ++j)
            {
                std::swap(lu(k, j), lu(max_row, j));
            }
        }

        // Step 3: Reject both exact singularity (pivot = 0, e.g. duplicate
        // rows) and near-singularity (pivot at noise level).
        real_type const pivot = abs(lu(k, k));
        const auto eps = std::numeric_limits<real_type>::epsilon();

        // Absolute threshold (~100 * machine eps); works for well-scaled inputs.
        // TODO: make it relative to matrix/column magnitude for better robustness.
        real_type const singular_tol = real_type(100) * eps;
        if (pivot <= singular_tol)
        {
            throw std::runtime_error("Lu::factorize: LU decomposition failed: singular or near-singular matrix.");
        }

        // Step 4: Gaussian elimination.  For each row below the pivot,
        // compute the multiplier l(i,k) = a(i,k) / a(k,k) and store it
        // in the lower triangle.  Then subtract l(i,k) * row_k from row_i
        // in the trailing submatrix (columns k+1..n-1).
        for (ssize_t i = k + 1; i < n; ++i)
        {
            lu(i, k) = lu(i, k) / lu(k, k); // multiplier stored in L
            for (ssize_t j = k + 1; j < n; ++j)
            {
                lu(i, j) = lu(i, j) - lu(i, k) * lu(k, j);
            }
        }
    }

    return {lu, piv};
}

template <typename T>
auto Lu<T>::forward_substitution(array_type const & lu, std::vector<ssize_t> const & piv, array_type const & b) -> array_type
{
    if (lu.ndim() != 2 || lu.shape(0) != lu.shape(1))
    {
        throw std::invalid_argument(std::format(
            "Lu::forward_substitution: lu must be a square 2D SimpleArray, but got shape {}",
            format_shape(lu)));
    }
    if (b.ndim() != 2 || b.shape(0) != lu.shape(0))
    {
        throw std::invalid_argument(std::format(
            "Lu::forward_substitution: b must be a 2D SimpleArray with first dimension matching lu, but got shape {}",
            format_shape(b)));
    }

    const auto m = static_cast<ssize_t>(lu.shape(0));
    const auto ncols = static_cast<ssize_t>(b.shape(1));
    small_vector<size_t> y_shape{static_cast<size_t>(m), static_cast<size_t>(ncols)};

    // Copy b into y so we can apply the permutation in-place.
    array_type y(y_shape);
    std::copy(b.begin(), b.end(), y.begin());

    // Apply the row permutation P to b.  The swaps are replayed in the
    // same order they were recorded during factorization (k = 0, 1, ...).
    for (ssize_t i = 0; i < m; ++i)
    {
        if (piv[i] != i)
        {
            for (ssize_t k = 0; k < ncols; ++k)
            {
                std::swap(y(i, k), y(piv[i], k));
            }
        }
    }

    // Solve Ly = Pb by forward substitution.  L is unit lower triangular
    // (diagonal is implicitly 1), so the formula for each element is:
    //   y(i, k) = y(i, k) - sum_{j=0}^{i-1} L(i,j) * y(j, k)
    for (ssize_t k = 0; k < ncols; ++k)
    {
        for (ssize_t i = 1; i < m; ++i)
        {
            value_type sum{0};
            for (ssize_t j = 0; j < i; ++j)
            {
                sum += lu(i, j) * y(j, k);
            }
            y(i, k) = y(i, k) - sum;
        }
    }

    return y;
}

template <typename T>
auto Lu<T>::backward_substitution(array_type const & lu, array_type const & y) -> array_type
{
    if (lu.ndim() != 2 || lu.shape(0) != lu.shape(1))
    {
        throw std::invalid_argument(std::format(
            "Lu::backward_substitution: lu must be a square 2D SimpleArray, but got shape {}",
            format_shape(lu)));
    }
    if (y.ndim() != 2 || y.shape(0) != lu.shape(0))
    {
        throw std::invalid_argument(std::format(
            "Lu::backward_substitution: y must be a 2D SimpleArray with first dimension matching lu, but got shape {}",
            format_shape(y)));
    }

    const auto m = static_cast<ssize_t>(lu.shape(0));
    const auto ncols = static_cast<ssize_t>(y.shape(1));
    small_vector<size_t> x_shape{static_cast<size_t>(m), static_cast<size_t>(ncols)};
    array_type x(x_shape);

    // Solve Ux = y by backward substitution.  U occupies the upper triangle
    // (including diagonal) of the combined lu matrix.  The formula is:
    //   x(i, k) = (y(i, k) - sum_{j=i+1}^{m-1} U(i,j) * x(j, k)) / U(i,i)
    for (ssize_t k = 0; k < ncols; ++k)
    {
        for (ssize_t i = m - 1; i >= 0; --i)
        {
            value_type sum{0};
            for (ssize_t j = i + 1; j < m; ++j)
            {
                sum += lu(i, j) * x(j, k);
            }
            x(i, k) = (y(i, k) - sum) / lu(i, i);
        }
    }

    return x;
}

template <typename T>
auto Lu<T>::solve(array_type const & a, array_type const & b) -> array_type
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1))
    {
        throw std::invalid_argument(std::format(
            "Lu::solve: a must be a square 2D SimpleArray, but got shape {}",
            format_shape(a)));
    }
    if (b.ndim() != 1 && b.ndim() != 2)
    {
        throw std::invalid_argument(std::format(
            "Lu::solve: b must be 1D or 2D, but got {}D",
            b.ndim()));
    }
    if (a.shape(0) != b.shape(0))
    {
        throw std::invalid_argument(std::format(
            "Lu::solve: dimension mismatch: a.shape[0]={}, b.shape[0]={}",
            a.shape(0),
            b.shape(0)));
    }

    // Factorize A into PA = LU, then solve by:
    //   1. Forward substitution:  Ly = Pb
    //   2. Backward substitution: Ux = y
    auto [lu, piv] = factorize(a);

    // If b is 1D, reshape to a column matrix (n, 1), solve, then reshape
    // the result back to 1D so the caller gets the same shape they passed in.
    bool const was_1d = (b.ndim() == 1);
    if (was_1d)
    {
        auto b_2d = b.reshape(small_vector<size_t>{b.shape(0), 1});
        auto y = forward_substitution(lu, piv, b_2d);
        auto x = backward_substitution(lu, y);
        return x.reshape(small_vector<size_t>{x.shape(0)});
    }
    else
    {
        auto y = forward_substitution(lu, piv, b);
        auto x = backward_substitution(lu, y);
        return x;
    }
}

template <typename T>
auto Lu<T>::inv(array_type const & a) -> array_type
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1))
    {
        throw std::invalid_argument(std::format(
            "Lu::inv: a must be a square 2D SimpleArray, but got shape {}",
            format_shape(a)));
    }

    // Matrix inversion is equivalent to solving AX = I for X, where I is the
    // n-by-n identity matrix.  Each column of X is the solution to A * x_j = e_j.
    const auto n = static_cast<ssize_t>(a.shape(0));
    auto identity = array_type::eye(n);

    auto [lu, piv] = factorize(a);
    auto y = forward_substitution(lu, piv, identity);
    auto x = backward_substitution(lu, y);
    return x;
}

} /* end namespace detail */

/**
 * Free function wrapper for LU factorization with partial pivoting.
 *
 * @param a  Square 2D SimpleArray.
 * @return   (lu_matrix, pivot_vector) pair.  See detail::Lu<T>::factorize().
 */
template <typename T>
std::pair<SimpleArray<T>, std::vector<ssize_t>> lu_factorization(SimpleArray<T> const & a)
{
    return detail::Lu<T>::factorize(a);
}

/**
 * Free function wrapper for solving Ax = b via LU decomposition.
 *
 * @param a  Square 2D SimpleArray (coefficient matrix).
 * @param b  1D or 2D SimpleArray (right-hand side).
 * @return   The solution x with the same shape as b.
 */
template <typename T>
SimpleArray<T> lu_solve(SimpleArray<T> const & a, SimpleArray<T> const & b)
{
    return detail::Lu<T>::solve(a, b);
}

/**
 * Free function wrapper for computing the matrix inverse via LU decomposition.
 *
 * @param a  Square 2D SimpleArray.
 * @return   The inverse A^(-1).
 */
template <typename T>
SimpleArray<T> lu_inv(SimpleArray<T> const & a)
{
    return detail::Lu<T>::inv(a);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
