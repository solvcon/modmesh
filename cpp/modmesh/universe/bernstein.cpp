/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/universe/bernstein.hpp>

namespace modmesh
{

double calc_bernstein_polynomial(double t, size_t i, size_t n)
{
    return detail::calc_bernstein_polynomial_impl<double>(t, i, n);
}

double interpolate_bernstein(double t, std::vector<double> const & values, size_t n)
{
    return detail::interpolate_bernstein_impl<double>(t, values, n);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
