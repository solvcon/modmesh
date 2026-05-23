/*
 * Copyright (c) 2026, An-Chi Liu <phy.tiger@gmail.com>
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

#include <modmesh/universe/pymod/universe_pymod.hpp> // Must be the first include.

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapViewTransform2dFp64
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

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
