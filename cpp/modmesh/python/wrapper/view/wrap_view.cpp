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

#include <modmesh/python/wrapper/view/view.hpp> // Must be the first include.

#include <modmesh/view/RMainWindow.hpp>
#include <modmesh/view/RStaticMesh.hpp>
#include <modmesh/view/R3DWidget.hpp>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapR3DWidget
    : public WrapBase<WrapR3DWidget, R3DWidget>
{

    friend root_base_type;

    WrapR3DWidget(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

#define DECL_QVECTOR3D_PROPERTY(NAME, GETTER, SETTER)          \
    .def_property(                                             \
        #NAME,                                                 \
        [](wrapped_type & self)                                \
        {                                                      \
            QVector3D const v = self.camera()->GETTER();       \
            return py::make_tuple(v.x(), v.y(), v.z());        \
        },                                                     \
        [](wrapped_type & self, std::vector<double> const & v) \
        {                                                      \
            double const x = v.at(0);                          \
            double const y = v.at(1);                          \
            double const z = v.at(2);                          \
            self.camera()->SETTER(QVector3D(x, y, z));         \
        })

        (*this)
            // clang-format off
            DECL_QVECTOR3D_PROPERTY(position, position, setPosition)
            DECL_QVECTOR3D_PROPERTY(up_vector, upVector, setUpVector)
            DECL_QVECTOR3D_PROPERTY(view_center, viewCenter, setViewCenter)
            // clang-format on
            ;

#undef DECL_QVECTOR3D_PROPERTY
    }

}; /* end class WrapR3DWidget */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRApplication
    : public WrapBase<WrapRApplication, RApplication>
{

    friend root_base_type;

    WrapRApplication(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "instance",
                [](py::object const &)
                {
                    return RApplication::instance();
                })
            .def_property_readonly(
                "viewer",
                [](wrapped_type & self)
                {
                    return self.main()->viewer();
                })
            //
            ;
    }

}; /* end class WrapRApplication */

void wrap_view(pybind11::module & mod)
{
    WrapR3DWidget::commit(mod, "R3DWidget", "R3DWidget");
    WrapRApplication::commit(mod, "RApplication", "RApplication");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
