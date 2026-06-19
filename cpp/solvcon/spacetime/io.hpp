#pragma once

/*
 * Copyright (c) 2018, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <iostream>

#include <solvcon/spacetime/core.hpp>
#include <solvcon/spacetime/kernel/linear_scalar.hpp>
#include <solvcon/spacetime/kernel/inviscid_burgers.hpp>

namespace solvcon
{

namespace spacetime
{

std::ostream & operator<<(std::ostream & os, const Grid & grid);
std::ostream & operator<<(std::ostream & os, const Field & sol);
std::ostream & operator<<(std::ostream & os, const Solver & sol);
std::ostream & operator<<(std::ostream & os, const Celm & elm);
std::ostream & operator<<(std::ostream & os, const Selm & elm);
std::ostream & operator<<(std::ostream & os, const InviscidBurgersSolver & sol);
std::ostream & operator<<(std::ostream & os, const InviscidBurgersCelm & elm);
std::ostream & operator<<(std::ostream & os, const InviscidBurgersSelm & elm);
std::ostream & operator<<(std::ostream & os, const LinearScalarSolver & sol);
std::ostream & operator<<(std::ostream & os, const LinearScalarCelm & elm);
std::ostream & operator<<(std::ostream & os, const LinearScalarSelm & elm);

} /* end namespace spacetime */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
