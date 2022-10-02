#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <iostream>

#include <modmesh/spacetime/core.hpp>
#include <modmesh/spacetime/kernel/linear_scalar.hpp>
#include <modmesh/spacetime/kernel/inviscid_burgers.hpp>
#include <modmesh/spacetime/kernel/BadEuler1DSolver.hpp>

namespace modmesh
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
std::ostream & operator<<(std::ostream & os, const BadEuler1DSolver & sol);

} /* end namespace spacetime */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
