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

#include <modmesh/view/wrap_view.hpp> // Must be the first include.
#include <modmesh/python/common.hpp>
#include <pybind11/stl.h>

#include <modmesh/view/view.hpp>

#include <QClipboard>

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

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRLine
    : public WrapBase<WrapRLine, RLine>
{

    friend root_base_type;

    WrapRLine(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                py::init(
                    [](float x0, float y0, float z0, float x1, float y1, float z1, uint8_t color_r, uint8_t color_g, uint8_t color_b)
                    {
                        auto * scene = RApplication::instance()->mainWindow()->viewer()->scene();
                        QVector3D v0(x0, y0, z0);
                        QVector3D v1(x1, y1, z1);
                        QColor color(color_r, color_g, color_b, 255);
                        auto * ret = new RLine(v0, v1, color, scene);
                        ret->addArrowHead(0.2f, 0.4f);
                        return ret;
                    }));
    }

}; /* end class WrapRLine */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRPythonConsoleDockWidget
    : public WrapBase<WrapRPythonConsoleDockWidget, RPythonConsoleDockWidget>
{

    friend root_base_type;

    WrapRPythonConsoleDockWidget(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

        (*this)
            .def_property(
                "command",
                [](wrapped_type const & self)
                {
                    return self.command().toStdString();
                },
                [](wrapped_type & self, std::string const & command)
                {
                    return self.setCommand(QString::fromStdString(command));
                })
            //
            ;
    }

}; /* end class WrapRPythonConsoleDockWidget */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRMainWindow
    : public WrapBase<WrapRMainWindow, RMainWindow>
{

    friend root_base_type;

    WrapRMainWindow(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .wrap_basic_qt()
            .def("clearApplications", &wrapped_type::clearApplications)
            .def(
                "addApplication",
                [](wrapped_type & self, std::string const & name)
                {
                    self.addApplication(QString::fromStdString(name));
                },
                py::arg("name"))
            .def_property_readonly(
                "viewer",
                [](wrapped_type & self)
                {
                    return self.viewer();
                })
            .def_property_readonly(
                "pycon",
                [](wrapped_type & self)
                {
                    return self.pycon();
                })
            //
            ;
    }

    wrapper_type & wrap_basic_qt()
    {
        namespace py = pybind11;

        (*this)
            .def("show", &wrapped_type::show)
            .def(
                "resize",
                [](wrapped_type & self, int w, int h)
                {
                    self.resize(w, h);
                },
                py::arg("w"),
                py::arg("h"))
            .def_property(
                "windowTitle",
                [](wrapped_type const & self)
                {
                    return self.windowTitle().toStdString();
                },
                [](wrapped_type & self, std::string const & name)
                {
                    self.setWindowTitle(QString::fromStdString(name));
                })
            //
            ;

        return *this;
    }

}; /* end class WrapRMainWindow */

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
                "mainWindow",
                [](wrapped_type & self)
                {
                    return self.mainWindow();
                })
            .def_property_readonly(
                "pycon",
                [](wrapped_type & self)
                {
                    return self.mainWindow()->pycon();
                })
            .def("setUp", &RApplication::setUp)
            .def(
                "exec",
                [](wrapped_type & self)
                {
                    return self.exec();
                })
            //
            ;
    }

}; /* end class WrapRApplication */

struct RApplicationProxy
{
};

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRApplicationProxy
    : public WrapBase<WrapRApplicationProxy, RApplicationProxy>
{

    friend root_base_type;

    WrapRApplicationProxy(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                "__getattr__",
                [](wrapped_type &, char const * name)
                {
                    py::object obj = py::cast(RApplication::instance());
                    obj = obj.attr(name);
                    return obj;
                })
            //
            ;
    }

}; /* end class WrapRApplicationProxy */

namespace detail
{

static void show_mark()
{
    RScene * scene = RApplication::instance()->mainWindow()->viewer()->scene();
    for (Qt3DCore::QNode * child : scene->childNodes())
    {
        if (typeid(*child) == typeid(RAxisMark))
        {
            child->deleteLater();
        }
    }
    new RAxisMark(scene);
}

static void update_appmesh(std::shared_ptr<StaticMesh> const & mesh)
{
    RScene * scene = RApplication::instance()->mainWindow()->viewer()->scene();
    for (Qt3DCore::QNode * child : scene->childNodes())
    {
        if (typeid(*child) == typeid(RStaticMesh))
        {
            child->deleteLater();
        }
    }
    new RStaticMesh(mesh, scene);
}

} /* end namespace detail */

void wrap_view(pybind11::module & mod)
{
    namespace py = pybind11;

    WrapR3DWidget::commit(mod, "R3DWidget", "R3DWidget");
    WrapRLine::commit(mod, "RLine", "RLine");
    WrapRPythonConsoleDockWidget::commit(mod, "RPythonConsoleDockWidget", "RPythonConsoleDockWidget");
    WrapRMainWindow::commit(mod, "RMainWindow", "RMainWindow");
    WrapRApplication::commit(mod, "RApplication", "RApplication");
    WrapRApplicationProxy::commit(mod, "RApplicationProxy", "RApplicationProxy");

    mod
        .def("show", &modmesh::python::detail::update_appmesh, py::arg("mesh"))
        .def("show_mark", &modmesh::python::detail::show_mark)
        .def(
            "clip_image",
            []()
            {
                R3DWidget * viewer = RApplication::instance()->mainWindow()->viewer();
                QClipboard * clipboard = QGuiApplication::clipboard();
                clipboard->setPixmap(viewer->grabPixmap());
            })
        .def(
            "save_image",
            [](std::string const & filename)
            {
                R3DWidget * viewer = RApplication::instance()->mainWindow()->viewer();
                viewer->grabPixmap().save(filename.c_str());
            },
            py::arg("filename"))
        //
        ;

    mod.attr("app") = RApplicationProxy();

    if (Toggle::instance().get_show_axis())
    {
        detail::show_mark();
    }
}

struct view_pymod_tag;

template <>
OneTimeInitializer<view_pymod_tag> & OneTimeInitializer<view_pymod_tag>::me()
{
    static OneTimeInitializer<view_pymod_tag> instance;
    return instance;
}

void initialize_view(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_view(mod);
    };

    OneTimeInitializer<view_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
