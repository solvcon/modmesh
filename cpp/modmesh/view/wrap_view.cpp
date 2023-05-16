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

#include <QPointer>
#include <QClipboard>

// Usually MODMESH_PYSIDE6_FULL is not defined unless for debugging.
#ifdef MODMESH_PYSIDE6_FULL
#include <pyside.h>
#else // MODMESH_PYSIDE6_FULL
namespace PySide
{
// The prototypes are taken from pyside.h
PyTypeObject * getTypeForQObject(const QObject * cppSelf);
PyObject * getWrapperForQObject(QObject * cppSelf, PyTypeObject * sbk_type);
QObject * convertToQObject(PyObject * object, bool raiseError);
} // end namespace PySide
#endif // MODMESH_PYSIDE6_FULL

namespace pybind11
{

namespace detail
{

template <typename type>
struct qt_type_caster
{
    // Adapted from PYBIND11_TYPE_CASTER.
protected:
    type * value;

public:
    template <typename T_, enable_if_t<std::is_same<type, remove_cv_t<T_>>::value, int> = 0>
    static handle cast(T_ * src, return_value_policy policy, handle parent)
    {
        if (!src)
            return none().release();
        if (policy == return_value_policy::take_ownership)
        {
            auto h = cast(std::move(*src), policy, parent);
            delete src;
            return h;
        }
        else
        {
            return cast(*src, policy, parent);
        }
    }
    operator type *() { return value; } /* NOLINT(bugprone-macro-parentheses) */
    operator type &() { return *value; } /* NOLINT(bugprone-macro-parentheses) */
    // Disable: operator type &&() && { return std::move(*value); } /* NOLINT(bugprone-macro-parentheses) */
    template <typename T_>
    using cast_op_type = pybind11::detail::movable_cast_op_type<T_>;
    // End adaptation from PYBIND11_TYPE_CASTER.

    bool load(handle src, bool)
    {
        if (!src)
        {
            return false;
        }

        QObject * q = PySide::convertToQObject(src.ptr(), /* raiseError */ true);
        if (!q)
        {
            return false;
        }

        value = qobject_cast<type *>(q);
        return true;
    }

    static handle cast(type * src, return_value_policy /* policy */, handle /* parent */)
    {
        PyObject * p = nullptr;
        PyTypeObject * to = PySide::getTypeForQObject(src);
        if (to)
        {
            p = PySide::getWrapperForQObject(src, to);
        }
        return pybind11::handle(p);
    }
};

#define QT_TYPE_CASTER(type, py_name)                      \
    template <>                                            \
    struct type_caster<type> : public qt_type_caster<type> \
    {                                                      \
        static constexpr auto name = py_name;              \
    }

QT_TYPE_CASTER(QWidget, _("QWidget"));
QT_TYPE_CASTER(QCoreApplication, _("QCoreApplication"));
QT_TYPE_CASTER(QApplication, _("QApplication"));
QT_TYPE_CASTER(QMainWindow, _("QMainWindow"));
QT_TYPE_CASTER(QMdiSubWindow, _("QMdiSubWindow"));

} /* end namespace detail */

} /* end namespace pybind11 */

PYBIND11_DECLARE_HOLDER_TYPE(T, QPointer<T>);

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapR3DWidget
    : public WrapBase<WrapR3DWidget, R3DWidget, QPointer<R3DWidget>>
{

    friend root_base_type;

    WrapR3DWidget(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

        (*this)
            .def_property_readonly("mesh", &wrapped_type::mesh)
            .def("updateMesh", &wrapped_type::updateMesh, py::arg("mesh"))
            .def("showMark", &wrapped_type::showMark)
            .def(
                "clipImage",
                [](wrapped_type & self)
                {
                    QClipboard * clipboard = QGuiApplication::clipboard();
                    clipboard->setPixmap(self.grabPixmap());
                })
            .def(
                "saveImage",
                [](wrapped_type & self, std::string const & filename)
                {
                    self.grabPixmap().save(filename.c_str());
                },
                py::arg("filename"))
            .def(
                "setCameraType",
                [](wrapped_type & self, std::string const & name)
                {
                    if (name == "orbit")
                    {
                        qDebug() << "Use Orbit Camera Controller";
                        self.scene()->setOrbitCameraController();
                        self.scene()->controller()->setCamera(self.camera());
                    }
                    else if (name == "fps")
                    {
                        qDebug() << "Use First Person Camera (fps) Controller";
                        self.scene()->setFirstPersonCameraController();
                        self.scene()->controller()->setCamera(self.camera());
                    }
                    else
                    {
                        qDebug() << "name needs to be either orbit or fps";
                    }
                },
                py::arg("name"))
            //
            ;

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

    static void updateMesh(wrapped_type & self, std::shared_ptr<StaticMesh> const & mesh)
    {
        RScene * scene = self.scene();
        for (Qt3DCore::QNode * child : scene->childNodes())
        {
            if (typeid(*child) == typeid(RStaticMesh))
            {
                child->deleteLater();
            }
        }
        new RStaticMesh(mesh, scene);
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
                    [](R3DWidget & w, float x0, float y0, float z0, float x1, float y1, float z1, uint8_t color_r, uint8_t color_g, uint8_t color_b)
                    {
                        auto * scene = w.scene();
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
            .def("writeToHistory", &wrapped_type::writeToHistory)
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
            .def_property(
                "python_redirect",
                [](wrapped_type const & self)
                {
                    return self.hasPythonRedirect();
                },
                [](wrapped_type & self, bool enabled)
                {
                    self.setPythonRedirect(enabled);
                })
            //
            ;
    }

}; /* end class WrapRPythonConsoleDockWidget */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRManager
    : public WrapBase<WrapRManager, RManager>
{

    friend root_base_type;

    WrapRManager(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "instance",
                [](py::object const &) -> wrapped_type &
                {
                    return RManager::instance();
                })
            .def_property_readonly_static(
                "core",
                [](py::object const &) -> QCoreApplication *
                {
                    return RManager::instance().core();
                })
            .def("setUp", &RManager::setUp)
            .def(
                "exec",
                [](wrapped_type & self)
                {
                    return self.core()->exec();
                })
            .wrap_widget()
            .wrap_app()
            .wrap_mainWindow()
            //
            ;
    }

    wrapper_type & wrap_widget()
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly(
                "pycon",
                [](wrapped_type & self)
                {
                    return self.pycon();
                })
            .def(
                "add3DWidget",
                [](wrapped_type & self)
                {
                    return self.add3DWidget();
                })
            //
            ;

        return *this;
    }

    wrapper_type & wrap_app()
    {
        namespace py = pybind11;

        (*this)
            .wrap_mainWindow()
            .def("clearApplications", &wrapped_type::clearApplications)
            .def(
                "addApplication",
                [](wrapped_type & self, std::string const & name)
                {
                    self.addApplication(QString::fromStdString(name));
                },
                py::arg("name"))
            //
            ;

        return *this;
    }

    wrapper_type & wrap_mainWindow()
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly(
                "mainWindow",
                [](wrapped_type & self) -> QMainWindow *
                {
                    return self.mainWindow();
                })
            .def(
                "show",
                [](wrapped_type & self)
                {
                    self.mainWindow()->show();
                })
            .def(
                "resize",
                [](wrapped_type & self, int w, int h)
                {
                    self.mainWindow()->resize(w, h);
                },
                py::arg("w"),
                py::arg("h"))
            .def(
                "addSubWindow",
                [](wrapped_type & self, QWidget * widget)
                {
                    QMdiSubWindow * subwin = self.addSubWindow(widget);
                    subwin->resize(300, 200);
                    subwin->setAttribute(Qt::WA_DeleteOnClose);
                    return subwin;
                },
                py::arg("widget"))
            .def_property(
                "windowTitle",
                [](wrapped_type & self)
                {
                    return self.mainWindow()->windowTitle().toStdString();
                },
                [](wrapped_type & self, std::string const & name)
                {
                    self.mainWindow()->setWindowTitle(QString::fromStdString(name));
                })
            //
            ;

        return *this;
    }

}; /* end class WrapRManager */

struct RManagerProxy
{
};

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRManagerProxy
    : public WrapBase<WrapRManagerProxy, RManagerProxy>
{

    friend root_base_type;

    WrapRManagerProxy(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                "__getattr__",
                [](wrapped_type &, char const * name)
                {
                    py::object obj = py::cast(RManager::instance());
                    obj = obj.attr(name);
                    return obj;
                })
            //
            ;
    }

}; /* end class WrapRManagerProxy */

void wrap_view(pybind11::module & mod)
{
    namespace py = pybind11;

    WrapR3DWidget::commit(mod, "R3DWidget", "R3DWidget");
    WrapRLine::commit(mod, "RLine", "RLine");
    WrapRPythonConsoleDockWidget::commit(mod, "RPythonConsoleDockWidget", "RPythonConsoleDockWidget");
    WrapRManager::commit(mod, "RManager", "RManager");
    WrapRManagerProxy::commit(mod, "RManagerProxy", "RManagerProxy");

    mod.attr("mgr") = RManagerProxy();
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

    if (Toggle::instance().solid().use_pyside())
    {
        try
        {
            pybind11::module::import("PySide6");
        }
        catch (const pybind11::error_already_set & e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    OneTimeInitializer<view_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
