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
#include <QMenu>

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
QT_TYPE_CASTER(QMenu, _("QMenu"));
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
            .def("updateWorld", &wrapped_type::updateWorld, py::arg("world"))
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
                "close_and_destroy",
                [](wrapped_base_type & self)
                { return self.closeAndDestroy(); })
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
            .def_property_readonly("camera", &wrapped_type::cameraController);
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
            .def_property_readonly("fileMenu", &wrapped_type::fileMenu)
            .def_property_readonly("viewMenu", &wrapped_type::viewMenu)
            .def_property_readonly("oneMenu", &wrapped_type::oneMenu)
            .def_property_readonly("meshMenu", &wrapped_type::meshMenu)
            .def_property_readonly("addonMenu", &wrapped_type::addonMenu)
            .def_property_readonly("windowMenu", &wrapped_type::windowMenu)
            .def(
                "quit",
                [](wrapped_type & self)
                {
                    self.quit();
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

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRCameraController
    : public WrapBase<WrapRCameraController, RCameraController>
{

    friend root_base_type;

    WrapRCameraController(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                "reset",
                [](wrapped_type & self)
                { self.reset(); })
            .def(
                "move",
                [](
                    wrapped_type & self,
                    float x,
                    float y,
                    float z,
                    float pitch,
                    float yaw,
                    bool left_mouse_button,
                    bool right_mouse_button,
                    bool alt_key,
                    bool shift_key)
                {
                    CameraInputState input{};
                    input.txAxisValue = x;
                    input.tyAxisValue = y;
                    input.tzAxisValue = z;
                    // yaw is for rotation around y-axis. horizontal movement of mouse rotates camera around y-axis
                    input.rxAxisValue = yaw;
                    // pitch is for rotation around x-axis. vertical movement of mouse rotates camera around x-axis
                    input.ryAxisValue = pitch;
                    input.leftMouseButtonActive = left_mouse_button;
                    input.rightMouseButtonActive = right_mouse_button;
                    input.altKeyActive = alt_key;
                    input.shiftKeyActive = shift_key;

                    constexpr float dt = 1.0f;
                    self.moveCamera(input, dt);
                },

                py::arg("x") = 0.f,
                py::arg("y") = 0.f,
                py::arg("z") = 0.f,
                py::arg("pitch") = 0.f,
                py::arg("yaw") = 0.f,
                py::arg("left_mouse_button") = false,
                py::arg("right_mouse_button") = false,
                py::arg("alt_key") = false,
                py::arg("shift_key") = false)
            .def_property_readonly(
                "view_vector",
                [](wrapped_base_type & self)
                {
                    const auto vector = self.viewVector();
                    return py::make_tuple(vector.x(), vector.y(), vector.z());
                });

#define MM_DECL_QVECTOR3D_PROPERTY(NAME, GETTER, SETTER)       \
    .def_property(                                             \
        #NAME,                                                 \
        [](wrapped_type & self)                                \
        {                                                      \
            QVector3D const v = self.GETTER();                 \
            return py::make_tuple(v.x(), v.y(), v.z());        \
        },                                                     \
        [](wrapped_type & self, std::vector<double> const & v) \
        {                                                      \
            double const x = v.at(0);                          \
            double const y = v.at(1);                          \
            double const z = v.at(2);                          \
            self.SETTER(QVector3D(x, y, z));                   \
        })

        (*this)
            // clang-format off
                    MM_DECL_QVECTOR3D_PROPERTY(position, position, setPosition)
                    MM_DECL_QVECTOR3D_PROPERTY(up_vector, upVector, setUpVector)
                    MM_DECL_QVECTOR3D_PROPERTY(view_center, viewCenter, setViewCenter)
                    MM_DECL_QVECTOR3D_PROPERTY(default_position, defaultPosition, setDefaultPosition)
                    MM_DECL_QVECTOR3D_PROPERTY(default_view_center, defaultViewCenter, setDefaultViewCenter)
                    MM_DECL_QVECTOR3D_PROPERTY(default_up_vector, defaultUpVector, setDefaultUpVector)
            // clang-format on
            ;
#undef MM_DECL_QVECTOR3D_PROPERTY

#define MM_DECL_FLOAT_PROPERTY(NAME, GETTER, SETTER) \
    .def_property(                                   \
        #NAME,                                       \
        [](wrapped_type & self)                      \
        {                                            \
            return self.GETTER();                    \
        },                                           \
        [](wrapped_type & self, float v)             \
        {                                            \
            self.SETTER(v);                          \
        })

        (*this)
            // clang-format off
                MM_DECL_FLOAT_PROPERTY(linear_speed, linearSpeed, setLinearSpeed)
                MM_DECL_FLOAT_PROPERTY(look_speed, lookSpeed, setLookSpeed)
                MM_DECL_FLOAT_PROPERTY(default_linear_speed, defaultLinearSpeed, setDefaultLinearSpeed)
                MM_DECL_FLOAT_PROPERTY(default_look_speed, defaultLookSpeed, setDefaultLookSpeed)
            // clang-format on
            ;
#undef MM_DECL_FLOAT_PROPERTY
    }
};

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
    WrapRCameraController::commit(mod, "RCameraController", "RCameraController");
    WrapRManager::commit(mod, "RManager", "RManager");
    WrapRManagerProxy::commit(mod, "RManagerProxy", "RManagerProxy");

    mod.attr("mgr") = RManagerProxy();

    try
    {
        // Creating module level variable to handle Qt MainWindow which is
        // created by c++ and registered it to Shiboken6 to prevent runtime
        // error occured.
        // RuntimeError:
        // Internal C++ object (PySide6.QtGui.QWindow) already deleted.
        // py::module::import("PySide6.QtWidgets");
        py::globals()["_mainWindow"] = RManager::instance().mainWindow();
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
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

    if (Toggle::instance().solid().use_pyside())
    {
        try
        {
            // Before using PySide6 api, the function signature need
            // to be imported or will get type error:
            // TypeError: Unable to convert function return value to a Python type!
            pybind11::module::import("PySide6.QtWidgets");
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
