/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/python/wrapper/spacetime/spacetime.hpp> // Must be the first include.

#include <modmesh/spacetime/spacetime.hpp>

#include <modmesh/python/wrapper/spacetime/wrap_spacetime.hpp>

namespace modmesh
{

namespace python
{

using namespace modmesh::spacetime; // NOLINT(google-build-using-namespace)

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapGrid
    : public WrapBase<WrapGrid, Grid, std::shared_ptr<Grid>>
{

    friend root_base_type;

    WrapGrid(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : root_base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .def(
                py::init(
                    [](real_type xmin, real_type xmax, size_t nelm)
                    {
                        return Grid::construct(xmin, xmax, nelm);
                    }),
                py::arg("xmin"),
                py::arg("xmax"),
                py::arg("nelm"))
            .def(
                py::init(
                    [](py::array_t<wrapped_type::value_type> & xloc)
                    {
                        return Grid::construct(makeSimpleArray(xloc));
                    }),
                py::arg("xloc"))
            .def("__str__", &detail::to_str<wrapped_type>)
            .def_property_readonly("xmin", &wrapped_type::xmin)
            .def_property_readonly("xmax", &wrapped_type::xmax)
            .def_property_readonly("ncelm", &wrapped_type::ncelm)
            .def_property_readonly("nselm", &wrapped_type::nselm)
            .def_property_readonly(
                "xcoord", static_cast<wrapped_type::array_type const & (wrapped_type::*)() const>(&wrapped_type::xcoord))
            .def_property_readonly_static("BOUND_COUNT", [](py::object const &)
                                          { return Grid::BOUND_COUNT; });
    }

}; /* end class WrapGrid */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapKernel
    : public WrapBase<WrapKernel, Kernel>
{

    friend root_base_type;

    WrapKernel(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : root_base_type(mod, pyname, clsdoc)
    {

        namespace py = pybind11;

        // clang-format off
#define DECL_ST_WRAP_CALCULATORS(NAME, TYPE) \
    .def_property \
    ( \
        #NAME \
      , [](wrapped_type & self) { return py::cpp_function(self.NAME()); } \
      , [](wrapped_type & self, typename wrapped_type::TYPE const & f) { self.NAME() = f; } \
    )
        (*this)
            DECL_ST_WRAP_CALCULATORS(xn_calc, calc_type1)
            DECL_ST_WRAP_CALCULATORS(xp_calc, calc_type1)
            DECL_ST_WRAP_CALCULATORS(tn_calc, calc_type1)
            DECL_ST_WRAP_CALCULATORS(tp_calc, calc_type1)
            DECL_ST_WRAP_CALCULATORS(so0p_calc, calc_type1)
            DECL_ST_WRAP_CALCULATORS(cfl_updater, calc_type2)
            .def("reset", &wrapped_type::reset)
        ;
#undef DECL_ST_WRAP_CALCULATORS
        // clang-format on
    }

}; /* end class WrapKernel */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapField
    : public WrapBase<WrapField, Field>
{

    friend root_base_type;

    WrapField(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : root_base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .def("__str__", &detail::to_str<wrapped_type>)
            .def_property_readonly(
                "grid",
                [](wrapped_type & self)
                { return self.grid().shared_from_this(); })
            .def_property_readonly("nvar", &wrapped_type::nvar)
            .def_property("time_increment", &wrapped_type::time_increment, &wrapped_type::set_time_increment)
            .def_property_readonly("dt", &wrapped_type::dt)
            .def_property_readonly("hdt", &wrapped_type::hdt)
            .def_property_readonly("qdt", &wrapped_type::qdt)
            .def(
                "celm",
                static_cast<Celm (wrapped_type::*)(int_type, bool)>(&wrapped_type::celm_at<Celm>),
                py::arg("ielm"),
                py::arg("odd_plane") = false)
            .def(
                "selm",
                static_cast<Selm (wrapped_type::*)(int_type, bool)>(&wrapped_type::selm_at<Selm>),
                py::arg("ielm"),
                py::arg("odd_plane") = false);
    }

}; /* end class WrapField */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapSolver
    : public WrapSolverBase<WrapSolver, Solver>
{

    using base_type = WrapSolverBase<WrapSolver, Solver>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;
    friend base_type::base_type;

    WrapSolver(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .def(
                py::init(
                    [](std::shared_ptr<Grid> const & grid, typename wrapped_type::value_type time_increment, size_t nvar)
                    {
                        return wrapped_type::construct(grid, time_increment, nvar);
                    }),
                py::arg("grid"),
                py::arg("time_increment"),
                py::arg("nvar"))
            // The kernel should only be exposed on the generic solver object.
            // C++-derived classes may use inline to avoid unnecessary function
            // calls.
            .def_property_readonly(
                "kernel",
                [](wrapped_type & self) -> Kernel &
                { return self.kernel(); });
    }

}; /* end class WrapSolver */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapCelm
    : public WrapCelmBase<WrapCelm, Celm>
{

    using base_type = WrapCelmBase<WrapCelm, Celm>;
    friend base_type::base_type::base_type;

    WrapCelm(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
    }

}; /* end class WrapCelm */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapSelm
    : public WrapSelmBase<WrapSelm, Selm>
{

    using base_type = WrapSelmBase<WrapSelm, Selm>;
    friend base_type::base_type::base_type;

    WrapSelm(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
    }

}; /* end class WrapSelm */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapInviscidBurgersSolver
    : public WrapSolverBase<WrapInviscidBurgersSolver, InviscidBurgersSolver>
{

    using base_type = WrapSolverBase<WrapInviscidBurgersSolver, InviscidBurgersSolver>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;
    friend base_type::base_type;

    WrapInviscidBurgersSolver(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .def(
                py::init(static_cast<std::shared_ptr<wrapped_type> (*)(
                             std::shared_ptr<Grid> const &, typename wrapped_type::value_type)>(&wrapped_type::construct)),
                py::arg("grid"),
                py::arg("time_increment"));
    }

}; /* end class WrapInviscidBurgersSolver */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapInviscidBurgersCelm
    : public WrapCelmBase<WrapInviscidBurgersCelm, InviscidBurgersCelm>
{

    using base_type = WrapCelmBase<WrapInviscidBurgersCelm, InviscidBurgersCelm>;
    friend base_type::base_type::base_type;

    WrapInviscidBurgersCelm(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
    }

}; /* end class WrapInviscidBurgersCelm */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapInviscidBurgersSelm
    : public WrapSelmBase<WrapInviscidBurgersSelm, InviscidBurgersSelm>
{

    using base_type = WrapSelmBase<WrapInviscidBurgersSelm, InviscidBurgersSelm>;
    friend base_type::base_type::base_type;

    WrapInviscidBurgersSelm(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
    }

}; /* end class WrapInviscidBurgersSelm */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapLinearScalarSolver
    : public WrapSolverBase<WrapLinearScalarSolver, LinearScalarSolver>
{

    using base_type = WrapSolverBase<WrapLinearScalarSolver, LinearScalarSolver>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;
    friend base_type::base_type;

    WrapLinearScalarSolver(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .def(
                py::init(static_cast<std::shared_ptr<wrapped_type> (*)(
                             std::shared_ptr<Grid> const &, typename wrapped_type::value_type)>(&wrapped_type::construct)),
                py::arg("grid"),
                py::arg("time_increment"));
    }

}; /* end class WrapLinearScalarSolver */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapLinearScalarCelm
    : public WrapCelmBase<WrapLinearScalarCelm, LinearScalarCelm>
{

    using base_type = WrapCelmBase<WrapLinearScalarCelm, LinearScalarCelm>;
    friend base_type::base_type::base_type;

    WrapLinearScalarCelm(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
    }

}; /* end class WrapLinearScalarCelm */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapLinearScalarSelm
    : public WrapSelmBase<WrapLinearScalarSelm, LinearScalarSelm>
{

    using base_type = WrapSelmBase<WrapLinearScalarSelm, LinearScalarSelm>;
    friend base_type::base_type::base_type;

    WrapLinearScalarSelm(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
    }

}; /* end class WrapLinearScalarSelm */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapBadEuler1DSolver
    : public WrapBase<WrapBadEuler1DSolver, BadEuler1DSolver, std::shared_ptr<BadEuler1DSolver>>
{

public:

    using base_type = WrapBase<WrapBadEuler1DSolver, BadEuler1DSolver, std::shared_ptr<BadEuler1DSolver>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapBadEuler1DSolver(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {

        namespace py = pybind11;

        (*this)
            .def(
                py::init(
                    [](std::shared_ptr<Grid> const & grid, double time_increment)
                    {
                        return wrapped_type::construct(grid, time_increment);
                    }),
                py::arg("grid"),
                py::arg("time_increment"))
            .def("__str__", &detail::to_str<wrapped_type>)
            .def_timed("clone", &wrapped_type::clone, py::arg("grid") = false)
            .def_property_readonly(
                "grid",
                [](wrapped_type & self)
                { return self.grid().shared_from_this(); })
            .def_property_readonly(
                "field", [](wrapped_type & self) -> auto & { return self.field(); });

        (*this)
            .def_group_array_getter()
            .def_group_array_setter()
            .def_group_march();
    }

    wrapper_type & def_group_array_getter()
    {
        namespace py = pybind11;

        (*this)
            .def_timed(
                "x",
                [](wrapped_type const & self, bool odd_plane)
                {
                    auto r = [](Selm const & s)
                    {
                        return s.x();
                    };
                    return loop_get_array(self, odd_plane, r);
                },
                py::arg("odd_plane") = false)
            .def_timed(
                "xctr",
                [](wrapped_type const & self, bool odd_plane)
                {
                    auto r = [](Selm const & s)
                    {
                        return s.xctr();
                    };
                    return loop_get_array(self, odd_plane, r);
                },
                py::arg("odd_plane") = false)
            .def_timed(
                "get_cfl",
                [](wrapped_type const & self, bool odd_plane)
                {
                    auto r = [](Selm const & s)
                    {
                        return s.cfl();
                    };
                    return loop_get_array(self, odd_plane, r);
                },
                py::arg("odd_plane") = false)
            .def_timed(
                "get_so0p",
                [](wrapped_type const & self, size_t iv, bool odd_plane)
                {
                    if (iv >= self.nvar())
                    {
                        throw std::out_of_range("get_so1(): out of nvar range");
                    }
                    auto r = [iv](Selm const & s)
                    {
                        return s.so0p(iv);
                    };
                    return loop_get_array(self, odd_plane, r);
                },
                py::arg("iv"),
                py::arg("odd_plane") = false)
            .def_timed(
                "get_so0",
                [](wrapped_type const & self, size_t iv, bool odd_plane)
                {
                    if (iv >= self.nvar())
                    {
                        throw std::out_of_range("get_so0(): out of nvar range");
                    }
                    auto r = [iv](Selm const & s)
                    {
                        return s.so0(iv);
                    };
                    return loop_get_array(self, odd_plane, r);
                },
                py::arg("iv"),
                py::arg("odd_plane") = false)
            .def_timed(
                "get_so1",
                [](wrapped_type const & self, size_t iv, bool odd_plane)
                {
                    if (iv >= self.nvar())
                    {
                        throw std::out_of_range("get_so1(): out of nvar range");
                    }
                    auto r = [iv](Selm const & s)
                    {
                        return s.so1(iv);
                    };
                    return loop_get_array(self, odd_plane, r);
                },
                py::arg("iv"),
                py::arg("odd_plane") = false);

        return *this;
    }

    template <typename R>
    static pybind11::array loop_get_array(
        wrapped_type const & self,
        bool odd_plane,
        R const & r)
    {
        const uint_type nselm =  static_cast<uint_type>(self.grid().nselm()) - static_cast<uint_type>(odd_plane);
        SimpleArray<double> ret(std::vector<size_t>{nselm});
        for (uint_type it = 0; it < nselm; ++it)
        {
            ret[it] = r(self.selm(it, odd_plane));
        }
        return to_ndarray(ret);
    }

    wrapper_type & def_group_array_setter()
    {
        namespace py = pybind11;

        (*this)
            .def(
                "set_so0",
                [](wrapped_type & self, size_t iv, py::array_t<double> & arr, bool odd_plane)
                {
                    auto r = [iv](Selm s, double v)
                    {
                        s.so0(iv) = v;
                    };
                    loop_set_array("set_so0", self, iv, makeSimpleArray(arr), odd_plane, r);
                },
                py::arg("iv"),
                py::arg("arr"),
                py::arg("odd_plane") = false)
            .def(
                "set_so1",
                [](wrapped_type & self, size_t iv, py::array_t<double> & arr, bool odd_plane)
                {
                    auto r = [iv](Selm s, double v)
                    {
                        s.so1(iv) = v;
                    };
                    loop_set_array("set_so1", self, iv, makeSimpleArray(arr), odd_plane, r);
                },
                py::arg("iv"),
                py::arg("arr"),
                py::arg("odd_plane") = false);

        return *this;
    }

    template <typename R>
    static void loop_set_array(
        std::string const & name,
        wrapped_type & self,
        size_t iv,
        SimpleArray<double> const & arr,
        bool odd_plane,
        R const & r)
    {
        if (iv >= self.nvar())
        {
            throw std::out_of_range(Formatter() << name << ": iv " << iv << " >= nvar " << self.nvar());
        }
        if (1 != arr.shape().size())
        {
            throw std::out_of_range(Formatter() << name << ": input not 1D");
        }
        const uint_type nselm = static_cast<uint_type>(self.grid().nselm() - odd_plane);
        if (nselm != arr.size())
        {
            throw std::out_of_range(Formatter() << name << ": arr size " << arr.size() << " != nselm " << nselm);
        }
        for (uint_type it = 0; it < nselm; ++it)
        {
            r(self.selm(it, odd_plane), arr[it]);
        }
    }

    wrapper_type & def_group_march()
    {
        namespace py = pybind11;

        (*this)
            .def_timed("update_cfl", &wrapped_type::update_cfl, py::arg("odd_plane"))
            .def_timed("march_half_so0", &wrapped_type::march_half_so0, py::arg("odd_plane"))
            .def_timed("treat_boundary_so0", &wrapped_type::treat_boundary_so0)
            .def_timed("treat_boundary_so1", &wrapped_type::treat_boundary_so1)
            .def_timed("setup_march", &wrapped_type::setup_march);

        (*this)
            .def_group_so1<0>()
            .def_group_so1<1>()
            .def_group_so1<2>();

        return *this;
    }

    template <size_t ALPHA>
    wrapper_type & def_group_so1()
    {
        // NOLINTNEXTLINE(misc-unused-alias-decls)
        namespace py = pybind11;

        (*this)
            .def_timed(
                (Formatter() << "march_half_so1_alpha" << ALPHA).str().c_str(),
                [](wrapped_type & self, bool odd_plane)
                {
                    return self.template march_half_so1_alpha<ALPHA>(odd_plane);
                },
                py::arg("odd_plane"))
            .def_timed(
                (Formatter() << "march_half1_alpha" << ALPHA).str().c_str(),
                [](wrapped_type & self)
                {
                    self.template march_half1_alpha<ALPHA>();
                })
            .def_timed(
                (Formatter() << "march_half2_alpha" << ALPHA).str().c_str(),
                [](wrapped_type & self)
                {
                    self.template march_half2_alpha<ALPHA>();
                })
            .def_timed(
                (Formatter() << "march_alpha" << ALPHA).str().c_str(),
                [](wrapped_type & self, size_t steps)
                {
                    self.template march_alpha<ALPHA>(steps);
                },
                py::arg("steps"));

        return *this;
    }

}; /* end class WrapBadEuler1DSolver */

template <typename WST, typename WCET, typename WSET>
void add_solver(pybind11::module & mod, const std::string & name, const std::string & desc)
{
    WST::commit(mod, (name + "Solver").c_str(), ("Solving algorithm of " + desc).c_str());
    WCET::commit(mod, (name + "Celm").c_str(), ("Conservation element of " + desc).c_str());
    WSET::commit(mod, (name + "Selm").c_str(), ("Solution element of " + desc).c_str());
}

void wrap_spacetime(pybind11::module & mod)
{
    mod.doc() = "_libst: One-dimensional space-time CESE method code";

    WrapGrid::commit(mod, "Grid", "Spatial grid data");
    WrapKernel::commit(mod, "Kernel", "Solution element calculation hooks");
    WrapField::commit(mod, "Field", "Solution data");

    add_solver<
        WrapSolver,
        WrapCelm,
        WrapSelm>(mod, "", "no equation");

    add_solver<
        WrapLinearScalarSolver,
        WrapLinearScalarCelm,
        WrapLinearScalarSelm>(mod, "LinearScalar", "a linear scalar equation");

    add_solver<
        WrapInviscidBurgersSolver,
        WrapInviscidBurgersCelm,
        WrapInviscidBurgersSelm>(mod, "InviscidBurgers", "the inviscid Burgers equation");

    WrapBadEuler1DSolver::commit(mod, "BadEuler1DSolver", "Solve the Euler equation (a bad one)");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
