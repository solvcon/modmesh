/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/universe/pymod/universe_pymod.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapViewTransform2dFp64
    : public WrapBase<WrapViewTransform2dFp64, ViewTransform2dFp64>
{

    friend root_base_type;

    WrapViewTransform2dFp64(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        // Constructors.
        (*this)
            .def(py::init<>())
            //
            ;

        // Properties and methods.
        (*this)
            .def_property(
                "pan_x",
                static_cast<double (wrapped_type::*)() const>(&wrapped_type::pan_x),
                &wrapped_type::set_pan_x)
            .def_property(
                "pan_y",
                static_cast<double (wrapped_type::*)() const>(&wrapped_type::pan_y),
                &wrapped_type::set_pan_y)
            .def_property(
                "zoom",
                static_cast<double (wrapped_type::*)() const>(&wrapped_type::zoom),
                &wrapped_type::set_zoom)
            .def("reset", &wrapped_type::reset)
            .def("pan", &wrapped_type::pan, py::arg("dx_screen"), py::arg("dy_screen"))
            .def(
                "zoom_at",
                &wrapped_type::zoom_at,
                py::arg("factor"),
                py::arg("anchor_screen_x"),
                py::arg("anchor_screen_y"))
            .def(
                "zoom_at_clamped",
                &wrapped_type::zoom_at_clamped,
                py::arg("factor"),
                py::arg("anchor_screen_x"),
                py::arg("anchor_screen_y"),
                py::arg("min_zoom"),
                py::arg("max_zoom"))
            .def(
                "screen_from_world",
                [](wrapped_type const & self, double world_x, double world_y)
                {
                    double screen_x = 0.0;
                    double screen_y = 0.0;
                    self.screen_from_world(world_x, world_y, screen_x, screen_y);
                    return py::make_tuple(screen_x, screen_y);
                },
                py::arg("world_x"),
                py::arg("world_y"))
            .def(
                "world_from_screen",
                [](wrapped_type const & self, double screen_x, double screen_y)
                {
                    double world_x = 0.0;
                    double world_y = 0.0;
                    self.world_from_screen(screen_x, screen_y, world_x, world_y);
                    return py::make_tuple(world_x, world_y);
                },
                py::arg("screen_x"),
                py::arg("screen_y"))
            //
            ;
    }

}; /* end class WrapViewTransform2dFp64 */

void wrap_view_transform2d(pybind11::module & mod)
{
    WrapViewTransform2dFp64::commit(mod, "ViewTransform2dFp64", "ViewTransform2dFp64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
